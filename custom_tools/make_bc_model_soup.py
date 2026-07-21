"""Average compatible BC checkpoints into one inference-cost-neutral model."""

import argparse
import copy
import json
from pathlib import Path


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredient", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float, default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    cli = parse_cli()
    import torch

    paths = [Path(item).expanduser().resolve() for item in cli.ingredient]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if cli.weight and len(cli.weight) != len(paths):
        raise ValueError("Provide one --weight per ingredient")
    weights = cli.weight or [1.0] * len(paths)
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Soup weights must be non-negative with a positive sum")
    weights = [weight / sum(weights) for weight in weights]
    output = Path(cli.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    checkpoints = [torch.load(str(path), map_location="cpu") for path in paths]
    states = [checkpoint["state_dict"] for checkpoint in checkpoints]
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("Ingredient state_dict keys do not match")
    averaged = {}
    for key in keys:
        tensors = [state[key] for state in states]
        reference = tensors[0]
        if any(tensor.shape != reference.shape or tensor.dtype != reference.dtype
               for tensor in tensors[1:]):
            raise ValueError("Incompatible tensor: {}".format(key))
        if reference.is_floating_point():
            value = torch.zeros_like(reference, dtype=torch.float64)
            for weight, tensor in zip(weights, tensors):
                value.add_(tensor.to(torch.float64), alpha=weight)
            averaged[key] = value.to(reference.dtype)
        else:
            if key.endswith("num_batches_tracked"):
                # This counter is not used by BatchNorm in eval mode.  Epochs
                # differ across ingredients, so preserve the largest count.
                averaged[key] = torch.stack(tensors).max(dim=0).values
            elif any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError("Non-floating buffer differs: {}".format(key))
            else:
                averaged[key] = reference.clone()

    result = copy.deepcopy(checkpoints[0])
    result["state_dict"] = averaged
    result["model_soup"] = {
        "method": "weighted_parameter_average",
        "ingredients": [str(path) for path in paths],
        "normalized_weights": weights,
        "same_inference_architecture": True,
    }
    # Optimizer state no longer corresponds to the averaged parameters.
    result.pop("optimizer_states", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, str(output))
    metadata = output.with_suffix(".json")
    with metadata.open("w", encoding="utf-8") as handle:
        json.dump(result["model_soup"], handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("MODEL_SOUP=READY")
    print("output={}".format(output))
    print("weights={}".format(weights))


if __name__ == "__main__":
    main()
