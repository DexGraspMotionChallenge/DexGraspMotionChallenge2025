"""CPU mathematical smoke test for the custom PPO implementation."""

import torch

from custom_tools.residual_ppo import (
    PPOConfig, ResidualActorCritic, RolloutStorage, ppo_update)


def main():
    torch.manual_seed(7)
    num_envs, rollout_steps = 4, 8
    model = ResidualActorCritic(32, 40, hidden_dims=(64, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig(update_epochs=2, minibatches=2)
    storage = RolloutStorage()
    actor_obs = torch.randn(num_envs, 32)
    critic_obs = torch.randn(num_envs, 40)
    for _ in range(rollout_steps):
        with torch.no_grad():
            action, log_prob, value = model.act(actor_obs, critic_obs)
        reward = torch.randn(num_envs)
        done = torch.zeros(num_envs, dtype=torch.bool)
        storage.add(actor_obs, critic_obs, action, log_prob, value, reward, done)
        actor_obs = torch.randn_like(actor_obs)
        critic_obs = torch.randn_like(critic_obs)
    with torch.no_grad():
        next_value = model.critic(critic_obs).squeeze(-1)
    batch = storage.finish(next_value, config)
    metrics = ppo_update(model, optimizer, batch, config)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert model.log_std.grad is not None
    print("[PASS] residual PPO math: {}".format(
        ", ".join("{}={:.4f}".format(k, v) for k, v in metrics.items())))

    gated = ResidualActorCritic(
        32, 40, hidden_dims=(64, 32), gate_dim=2, initial_gate=0.1)
    with torch.no_grad():
        action, log_prob, value = gated.act(
            torch.zeros(num_envs, 32), torch.zeros(num_envs, 40),
            deterministic=True)
    gates = 0.5 * (action[:, -2:] + 1.0)
    assert action.shape == (num_envs, 30)
    assert torch.allclose(gates, torch.full_like(gates, 0.1), atol=2e-3)
    assert torch.isfinite(log_prob).all() and torch.isfinite(value).all()
    print("[PASS] gated residual policy: action_dim=30, initial_gate={:.3f}".format(
        gates.mean().item()))

    gated_optimizer = torch.optim.Adam(gated.parameters(), lr=3e-4)
    anchored_config = PPOConfig(
        update_epochs=1, minibatches=2,
        anchor_effective_residual_coef=0.1, anchor_gate_coef=0.01)
    anchored_storage = RolloutStorage()
    actor_obs = torch.randn(num_envs, 32)
    critic_obs = torch.randn(num_envs, 40)
    for _ in range(rollout_steps):
        with torch.no_grad():
            action, log_prob, value = gated.act(actor_obs, critic_obs)
        anchored_storage.add(
            actor_obs, critic_obs, action, log_prob, value,
            torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool))
        actor_obs = torch.randn_like(actor_obs)
        critic_obs = torch.randn_like(critic_obs)
    with torch.no_grad():
        next_value = gated.critic(critic_obs).squeeze(-1)
    anchored_batch = anchored_storage.finish(
        next_value, anchored_config,
        anchor_env_mask=torch.tensor([True, True, False, False]))
    anchored_metrics = ppo_update(
        gated, gated_optimizer, anchored_batch, anchored_config)
    assert anchored_metrics["anchor_sample_fraction"] > 0
    assert anchored_metrics["anchor_gate_loss"] >= 0
    assert anchored_metrics["anchor_effective_residual_loss"] >= 0
    print("[PASS] behavior anchor losses: effective={:.6f}, gate={:.6f}".format(
        anchored_metrics["anchor_effective_residual_loss"],
        anchored_metrics["anchor_gate_loss"]))


if __name__ == "__main__":
    main()
