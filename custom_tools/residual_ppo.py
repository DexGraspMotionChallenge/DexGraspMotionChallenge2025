"""Small PPO implementation for the custom frozen-BC residual policy.

This module deliberately has no Isaac Gym imports, so its policy and update
math can be checked on CPU before a simulator is created.
"""

from dataclasses import dataclass

import torch
from torch import nn


def _mlp(input_dim, hidden_dims, output_dim):
    # Per-sample normalization prevents the large DexRep vector from driving
    # an untrained residual actor into saturated tanh actions.
    layers = [nn.LayerNorm(input_dim, elementwise_affine=False)]
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), nn.ELU()))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualActorCritic(nn.Module):
    """Separate actor and privileged critic used by residual PPO."""

    def __init__(self, actor_obs_dim, critic_obs_dim, action_dim=28,
                 hidden_dims=(512, 256), init_std=0.25, gate_dim=0,
                 initial_gate=0.1):
        super().__init__()
        if gate_dim < 0:
            raise ValueError("gate_dim cannot be negative")
        if gate_dim and not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be strictly between zero and one")
        self.residual_action_dim = int(action_dim)
        self.gate_dim = int(gate_dim)
        self.action_dim = self.residual_action_dim + self.gate_dim
        self.initial_gate = float(initial_gate)
        self.actor = _mlp(actor_obs_dim, hidden_dims, self.action_dim)
        self.critic = _mlp(critic_obs_dim, hidden_dims, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(torch.log(torch.tensor(init_std)))))
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        if self.gate_dim:
            # The last outputs are tanh-squashed and then mapped from [-1, 1]
            # to [0, 1].  This bias therefore starts both gates near the
            # requested value while keeping them inside PPO's sampled action.
            gate_pre_tanh = torch.atanh(torch.tensor(
                2.0 * self.initial_gate - 1.0))
            with torch.no_grad():
                self.actor[-1].bias[-self.gate_dim:].fill_(gate_pre_tanh.item())

    def distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @staticmethod
    def _squashed_log_prob(distribution, pre_tanh, action):
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)

    def act(self, actor_obs, critic_obs, deterministic=False):
        distribution = self.distribution(actor_obs)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, action)
        value = self.critic(critic_obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, actor_obs, critic_obs, action):
        action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(action)
        distribution = self.distribution(actor_obs)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, action)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(critic_obs).squeeze(-1)
        return log_prob, entropy, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatches: int = 4
    target_kl: float = 0.03
    advantage_normalization: str = "global"
    min_advantage_std: float = 1.0
    advantage_clip: float = 5.0
    anchor_effective_residual_coef: float = 0.0
    anchor_gate_coef: float = 0.0


def normalize_advantages(advantages, mode="global", group_ids=None,
                         min_std=1.0, clip_value=5.0):
    """Normalize advantages globally or per task/category.

    A minimum divisor of one prevents tiny, noisy category advantages from
    being amplified merely to obtain unit variance.
    """
    if mode not in ("global", "category"):
        raise ValueError("advantage normalization must be global or category")
    normalized = torch.empty_like(advantages)
    if mode == "global":
        groups = torch.zeros_like(advantages, dtype=torch.long)
    else:
        if group_ids is None or group_ids.shape != advantages.shape:
            raise ValueError("category normalization needs one group id per advantage")
        groups = group_ids.long()
    stats = {}
    for group in torch.unique(groups, sorted=True):
        mask = groups == group
        values = advantages[mask]
        mean = values.mean()
        std = values.std(unbiased=False)
        divisor = torch.clamp(std, min=float(min_std))
        transformed = ((values - mean) / divisor).clamp(
            -float(clip_value), float(clip_value))
        normalized[mask] = transformed
        stats[int(group.item())] = {
            "raw_mean": float(mean.item()),
            "raw_std": float(std.item()),
            "divisor": float(divisor.item()),
            "normalized_mean": float(transformed.mean().item()),
            "normalized_std": float(transformed.std(unbiased=False).item()),
            "normalized_abs_mean": float(transformed.abs().mean().item()),
            "sample_count": int(mask.sum().item()),
        }
    return normalized, stats


class RolloutStorage:
    """Time-major rollout storage kept on the simulation device."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.actor_obs = []
        self.critic_obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, actor_obs, critic_obs, actions, log_probs, values, rewards, dones):
        for target, value in (
                (self.actor_obs, actor_obs), (self.critic_obs, critic_obs),
                (self.actions, actions), (self.log_probs, log_probs),
                (self.values, values), (self.rewards, rewards),
                (self.dones, dones)):
            target.append(value.detach().clone())

    def finish(self, next_value, config, advantage_groups=None,
               anchor_env_mask=None):
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones).float()
        values = torch.stack(self.values)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)
        bootstrap = next_value
        for step in reversed(range(rewards.shape[0])):
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + config.gamma * bootstrap * nonterminal - values[step]
            gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
            advantages[step] = gae
            bootstrap = values[step]
        returns = advantages + values
        flat_advantages = advantages.flatten()
        flat_groups = None
        if config.advantage_normalization == "category":
            if advantage_groups is None or advantage_groups.numel() != rewards.shape[1]:
                raise ValueError("Provide one category id per environment")
            flat_groups = advantage_groups.repeat(rewards.shape[0])
        normalized_advantages, advantage_stats = normalize_advantages(
            flat_advantages,
            mode=config.advantage_normalization,
            group_ids=flat_groups,
            min_std=config.min_advantage_std,
            clip_value=config.advantage_clip,
        )
        batch = {
            "actor_obs": torch.cat(self.actor_obs),
            "critic_obs": torch.cat(self.critic_obs),
            "actions": torch.cat(self.actions),
            "old_log_probs": torch.cat(self.log_probs),
            "old_values": values.flatten(),
            "advantages": normalized_advantages,
            "returns": returns.flatten(),
            "advantage_group_stats": advantage_stats,
        }
        if anchor_env_mask is not None:
            if anchor_env_mask.numel() != rewards.shape[1]:
                raise ValueError("Provide one anchor flag per environment")
            batch["anchor_mask"] = anchor_env_mask.bool().repeat(rewards.shape[0])
        return batch


def ppo_update(model, optimizer, batch, config):
    batch_size = batch["actions"].shape[0]
    minibatch_size = max(1, batch_size // config.minibatches)
    totals = {name: 0.0 for name in (
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        "anchor_effective_residual_loss", "anchor_gate_loss",
        "anchor_sample_fraction")}
    updates = 0
    early_stopped = False

    for _ in range(config.update_epochs):
        permutation = torch.randperm(batch_size, device=batch["actions"].device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start:start + minibatch_size]
            log_prob, entropy, value = model.evaluate(
                batch["actor_obs"][indices], batch["critic_obs"][indices],
                batch["actions"][indices])
            log_ratio = log_prob - batch["old_log_probs"][indices]
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            if updates > 0 and approx_kl.item() > config.target_kl:
                early_stopped = True
                break
            advantage = batch["advantages"][indices]
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - batch["returns"][indices]).square().mean()
            entropy_mean = entropy.mean()
            anchor_effective_loss = policy_loss.new_zeros(())
            anchor_gate_loss = policy_loss.new_zeros(())
            anchor_fraction = 0.0
            anchor_mask = batch.get("anchor_mask")
            if anchor_mask is not None:
                minibatch_anchor_mask = anchor_mask[indices]
                anchor_fraction = float(
                    minibatch_anchor_mask.float().mean().item())
                if minibatch_anchor_mask.any():
                    mean_action = torch.tanh(model.distribution(
                        batch["actor_obs"][indices][minibatch_anchor_mask]).mean)
                    residual = mean_action[:, :model.residual_action_dim]
                    if model.gate_dim:
                        gates = 0.5 * (mean_action[:, -model.gate_dim:] + 1.0)
                        wrist = residual[:, :6] * gates[:, :1]
                        fingers = residual[:, 6:] * gates[:, 1:2]
                        effective = torch.cat((wrist, fingers), dim=-1)
                        anchor_gate_loss = gates.square().mean()
                    else:
                        effective = residual
                    anchor_effective_loss = effective.square().mean()
            loss = (
                policy_loss + config.value_coef * value_loss
                - config.entropy_coef * entropy_mean
                + config.anchor_effective_residual_coef * anchor_effective_loss
                + config.anchor_gate_coef * anchor_gate_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                totals["approx_kl"] += approx_kl.item()
                totals["clip_fraction"] += (
                    (ratio - 1.0).abs() > config.clip_ratio).float().mean().item()
                totals["anchor_effective_residual_loss"] += (
                    anchor_effective_loss.item())
                totals["anchor_gate_loss"] += anchor_gate_loss.item()
                totals["anchor_sample_fraction"] += anchor_fraction
                updates += 1
        if early_stopped:
            break

    metrics = {name: value / updates for name, value in totals.items()}
    metrics["ppo_updates"] = updates
    metrics["kl_early_stop"] = int(early_stopped)
    return metrics
