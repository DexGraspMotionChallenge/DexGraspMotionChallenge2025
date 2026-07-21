"""Frozen behavior-cloning policy plus a bounded residual-control wrapper."""

import torch

from custom_tools.evaluation_loop import obs_process
from custom_tools.reward_shaping import (
    LIFT_PROGRESS_MODES, compute_lift_progress)


class ResidualDexGraspEnv:
    """Expose actor/critic observations and a shaped PPO training reward.

    The wrapped challenge task remains responsible for physics and for its
    original ``successes`` flag.  This class never redefines that flag.
    """

    def __init__(self, task, bc_model, horizon=122, history_frames=3,
                 wrist_residual_scale=0.05, finger_residual_scale=0.10,
                 contact_force_threshold=0.5, reset_settle_steps=4,
                 reward_config=None, gated_residual=False):
        if history_frames < 1:
            raise ValueError("history_frames must be at least one")
        self.task = task
        self.bc_model = bc_model.eval()
        for parameter in self.bc_model.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device(task.device)
        self.num_envs = task.num_envs
        self.action_dim = 28
        self.gated_residual = bool(gated_residual)
        self.gate_dim = 2 if self.gated_residual else 0
        self.policy_action_dim = self.action_dim + self.gate_dim
        self.horizon = int(horizon)
        self.history_frames = int(history_frames)
        self.contact_force_threshold = float(contact_force_threshold)
        self.reset_settle_steps = int(reset_settle_steps)
        if self.reset_settle_steps < 0:
            raise ValueError("reset_settle_steps must be non-negative")
        default_reward_config = {
            "approach_progress": 10.0,
            # Zero preserves the original reward.  A positive threshold makes
            # approach progress active only until that many fingertip contacts
            # have been reached once in the current episode.
            "approach_until_contact_count": 0,
            "contact_gain": 0.25,
            "lift_progress": 40.0,
            "lift_progress_mode": "signed_delta",
            "milestone_heights_m": [0.02, 0.05, 0.10, 0.20],
            "milestone_rewards": [0.5, 0.5, 0.5, 0.5],
            "official_success_bonus": 10.0,
            "drop_penalty": 10.0,
            "fly_penalty": 10.0,
            "normalized_residual_penalty": 0.01,
            "normalized_residual_smoothness_penalty": 0.01,
            "gate_penalty": 0.0,
        }
        self.reward_config = default_reward_config
        if reward_config is not None:
            self.reward_config.update(dict(reward_config))
        if (len(self.reward_config["milestone_heights_m"])
                != len(self.reward_config["milestone_rewards"])):
            raise ValueError("milestone heights and rewards must have equal length")
        if self.reward_config["lift_progress_mode"] not in LIFT_PROGRESS_MODES:
            raise ValueError(
                "lift_progress_mode must be one of {}, got {!r}".format(
                    LIFT_PROGRESS_MODES,
                    self.reward_config["lift_progress_mode"]))
        approach_contact_count = int(
            self.reward_config["approach_until_contact_count"])
        if not 0 <= approach_contact_count <= 5:
            raise ValueError(
                "approach_until_contact_count must be between zero and five")
        self.reward_config["approach_until_contact_count"] = (
            approach_contact_count)
        self.residual_scale = torch.cat((
            torch.full((6,), float(wrist_residual_scale), device=self.device),
            torch.full((22,), float(finger_residual_scale), device=self.device),
        ))
        self.prop_dim = int(task.cfg["env"]["obs_dim"]["prop"])
        self.processed_obs_dim = 2460 if self.prop_dim == 100 else None
        self._prop_history = None
        self._action_history = None
        self._bc_action = None
        self._actor_obs = None
        self._critic_obs = None
        self._initial_height = None
        self._previous_palm_distance = None
        self._previous_finger_distance = None
        self._previous_height = None
        self._maximum_height = None
        self._previous_contacts = None
        self._previous_final_action = None
        self._previous_normalized_residual = None
        self._ever_success = None
        self._crossed_milestones = None
        self._approach_phase_complete = None
        self._settle_steps_remaining = None

    @property
    def actor_obs_dim(self):
        return self._actor_obs.shape[-1]

    @property
    def critic_obs_dim(self):
        return self._critic_obs.shape[-1]

    @torch.no_grad()
    def _processed_observation(self):
        processed = obs_process(self.task.obs_buf, pro_dim=self.prop_dim)
        if self.processed_obs_dim is not None and processed.shape[-1] != self.processed_obs_dim:
            raise RuntimeError(
                "Expected processed observation {}, got {}".format(
                    self.processed_obs_dim, processed.shape[-1]))
        return torch.nan_to_num(processed, nan=0.0, posinf=10.0, neginf=-10.0)

    @torch.no_grad()
    def _predict_bc_action(self, processed_obs):
        return self.bc_model.model.act_inference(processed_obs).clamp(-1.0, 1.0)

    def _distances_and_contacts(self):
        palm_distance = torch.linalg.vector_norm(
            self.task.object_pos - self.task.right_hand_pos, dim=-1)
        fingertips = torch.stack((
            self.task.right_hand_ff_pos, self.task.right_hand_mf_pos,
            self.task.right_hand_rf_pos, self.task.right_hand_lf_pos,
            self.task.right_hand_th_pos), dim=1)
        finger_distance = torch.linalg.vector_norm(
            fingertips - self.task.object_pos[:, None, :], dim=-1).mean(dim=-1)
        force = self.task.vec_sensor_tensor.view(self.num_envs, 5, 6)[..., :3]
        contacts = (
            torch.linalg.vector_norm(force, dim=-1) > self.contact_force_threshold
        ).float().sum(dim=-1)
        return palm_distance, finger_distance, contacts

    def _privileged_state(self):
        fingertips = torch.cat((
            self.task.right_hand_ff_pos, self.task.right_hand_mf_pos,
            self.task.right_hand_rf_pos, self.task.right_hand_lf_pos,
            self.task.right_hand_th_pos), dim=-1)
        force_norms = torch.linalg.vector_norm(
            self.task.vec_sensor_tensor.view(self.num_envs, 5, 6)[..., :3], dim=-1)
        height_delta = (self.task.object_pos[:, 2] - self._initial_height).unsqueeze(-1)
        goal_delta = self.task.goal_pos - self.task.object_pos
        return torch.cat((
            self.task.object_pos, self.task.object_rot,
            self.task.object_linvel, self.task.object_angvel,
            self.task.right_hand_pos, fingertips, force_norms,
            height_delta, goal_delta, self.task.shadow_hand_dof_vel,
        ), dim=-1)

    def _assemble_observations(self, processed_obs, bc_action):
        actor_obs = torch.cat((
            processed_obs, bc_action,
            self._prop_history.reshape(self.num_envs, -1),
            self._action_history.reshape(self.num_envs, -1),
        ), dim=-1)
        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        critic_obs = torch.cat((actor_obs, self._privileged_state()), dim=-1)
        self._actor_obs = actor_obs.clamp(-10.0, 10.0)
        self._critic_obs = torch.nan_to_num(
            critic_obs, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    @torch.no_grad()
    def reset(self):
        # Match the verified official VecTaskPython.reset path.  The id=-1
        # settle step initializes ``task.actions`` and advances ten physics
        # frames after resetting all environments.
        # Training episodes themselves must start at progress zero.  The
        # challenge config's one-time random_time option is useful for some
        # data extraction workflows but would create artificial early PPO
        # timeouts when this wrapper adds an explicit horizon.
        self.task.random_time = False
        self.task.reset_buf[:] = 1
        self.task.progress_buf[:] = 0
        zero_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device)
        self.task.step(zero_actions, id=-1)
        self.task.progress_buf[:] = 0
        processed = self._processed_observation()
        bc_action = self._predict_bc_action(processed)
        prop = processed[:, :self.prop_dim]
        self._prop_history = prop[:, None, :].repeat(1, self.history_frames, 1)
        self._action_history = bc_action[:, None, :].repeat(
            1, max(0, self.history_frames - 1), 1)
        self._initial_height = self.task.object_pos[:, 2].clone()
        palm, finger, contacts = self._distances_and_contacts()
        self._previous_palm_distance = palm
        self._previous_finger_distance = finger
        self._previous_height = self.task.object_pos[:, 2].clone()
        self._maximum_height = self.task.object_pos[:, 2].clone()
        self._previous_contacts = contacts
        self._previous_final_action = bc_action.clone()
        self._ever_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_normalized_residual = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device)
        self._crossed_milestones = torch.zeros(
            (self.num_envs, len(self.reward_config["milestone_heights_m"])),
            dtype=torch.bool, device=self.device)
        self._approach_phase_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._settle_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self._bc_action = bc_action
        self._assemble_observations(processed, bc_action)
        return self._actor_obs, self._critic_obs

    @torch.no_grad()
    def reset_done(self, done):
        """Reset completed environments before the next policy action.

        Isaac Gym's original task resets inside ``pre_physics_step`` and would
        therefore discard the next PPO action.  Performing the indexed reset
        here preserves the standard vector-environment transition contract.
        """
        env_ids = done.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return self._actor_obs, self._critic_obs
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        self.task.reset(env_ids, empty)
        self.task.actions[env_ids] = 0.0
        self.task.compute_observations()
        processed = self._processed_observation()
        bc_action = self._predict_bc_action(processed)
        prop = processed[:, :self.prop_dim]
        self._prop_history[env_ids] = prop[env_ids, None, :].repeat(
            1, self.history_frames, 1)
        if self.history_frames > 1:
            self._action_history[env_ids] = bc_action[env_ids, None, :].repeat(
                1, self.history_frames - 1, 1)
        self._initial_height[env_ids] = self.task.object_pos[env_ids, 2]
        palm, finger, contacts = self._distances_and_contacts()
        self._previous_palm_distance[env_ids] = palm[env_ids]
        self._previous_finger_distance[env_ids] = finger[env_ids]
        self._previous_height[env_ids] = self.task.object_pos[env_ids, 2]
        self._maximum_height[env_ids] = self.task.object_pos[env_ids, 2]
        self._previous_contacts[env_ids] = contacts[env_ids]
        self._previous_final_action[env_ids] = bc_action[env_ids]
        self._previous_normalized_residual[env_ids] = 0.0
        self._ever_success[env_ids] = False
        self._crossed_milestones[env_ids] = False
        self._approach_phase_complete[env_ids] = False
        self._settle_steps_remaining[env_ids] = self.reset_settle_steps
        self._bc_action = bc_action
        self._assemble_observations(processed, bc_action)
        return self._actor_obs, self._critic_obs

    def _reward_and_done(self, effective_residual, final_action, gates):
        palm, finger, contacts = self._distances_and_contacts()
        height = self.task.object_pos[:, 2]
        settling = self._settle_steps_remaining > 0
        active_float = (~settling).float()
        approach = float(self.reward_config["approach_progress"]) * (
            self._previous_palm_distance - palm
            + self._previous_finger_distance - finger).clamp(-0.05, 0.05)
        approach_contact_count = self.reward_config[
            "approach_until_contact_count"]
        if approach_contact_count > 0:
            # Keep the transition that first forms the grasp in the approach
            # phase, then permanently disable approach reward for this episode.
            approach *= (~self._approach_phase_complete).float()
            reached_grasp_phase = (
                (contacts >= approach_contact_count) & ~settling)
            self._approach_phase_complete |= reached_grasp_phase
        approach *= active_float
        contact_gain = float(self.reward_config["contact_gain"]) * (
            contacts - self._previous_contacts).clamp(-2.0, 2.0)
        contact_gain *= active_float
        lift_progress, updated_maximum_height = compute_lift_progress(
            height, self._previous_height, self._maximum_height,
            mode=self.reward_config["lift_progress_mode"], clamp_m=0.03)
        lift = float(self.reward_config["lift_progress"]) * lift_progress
        lift *= active_float

        height_delta = height - self._initial_height
        milestone_levels = torch.tensor(
            self.reward_config["milestone_heights_m"], device=self.device)
        milestone_rewards = torch.tensor(
            self.reward_config["milestone_rewards"], device=self.device)
        reached = (
            height_delta[:, None] >= milestone_levels[None, :]
        ) & (~settling[:, None])
        new_milestones = reached & ~self._crossed_milestones
        milestone = (new_milestones.float() * milestone_rewards[None, :]).sum(dim=-1)
        self._crossed_milestones |= reached

        official_success = (self.task.successes > 0) & ~settling
        new_success = official_success & ~self._ever_success
        success = float(self.reward_config["official_success_bonus"]) * new_success.float()
        self._ever_success |= official_success

        drop = (height < (self._initial_height - 0.03)) & ~settling
        fly = (
            (self.task.object_pos[:, 0].abs() > 1.5)
            | (self.task.object_pos[:, 1].abs() > 1.5)
            | (height >= 2.0)) & ~settling
        failure = (
            -float(self.reward_config["drop_penalty"]) * drop.float()
            -float(self.reward_config["fly_penalty"]) * fly.float())
        residual_cost = -float(
            self.reward_config["normalized_residual_penalty"]
        ) * effective_residual.square().mean(dim=-1)
        residual_cost *= active_float
        smoothness = -float(
            self.reward_config["normalized_residual_smoothness_penalty"]
        ) * (effective_residual - self._previous_normalized_residual).square().mean(dim=-1)
        smoothness *= active_float
        gate_cost = -float(self.reward_config["gate_penalty"]) * gates.mean(dim=-1)
        gate_cost *= active_float
        reward = (
            approach + contact_gain + lift + milestone + success + failure
            + residual_cost + smoothness + gate_cost)
        timeout = self.task.progress_buf >= self.horizon
        done = new_success | drop | fly | timeout
        if settling.any():
            # The official full reset advances ten physics frames before the
            # episode baseline is measured.  Indexed resets cannot advance
            # only selected Isaac Gym environments, so treat the equivalent
            # local settling transitions as neutral and refresh the baseline.
            self._initial_height[settling] = height[settling]
            updated_maximum_height[settling] = height[settling]
            self.task.progress_buf[settling] = 0
            self.task.successes[settling] = 0
            self._ever_success[settling] = False
            self._crossed_milestones[settling] = False
            self._approach_phase_complete[settling] = False
            self._settle_steps_remaining[settling] -= 1
            height_delta = torch.where(
                settling, torch.zeros_like(height_delta), height_delta)
            done = done & ~settling
        terms = {
            "reward": reward,
            "approach": approach,
            "contact": contact_gain,
            "lift": lift,
            "milestone": milestone,
            "success_bonus": success,
            "failure_penalty": failure,
            "residual_penalty": residual_cost,
            "smoothness_penalty": smoothness,
            "gate_penalty": gate_cost,
            "wrist_gate": gates[:, 0],
            "finger_gate": gates[:, 1],
            "official_success": official_success.float(),
            "height_delta": height_delta,
            "contact_count": contacts,
            "approach_phase_complete": self._approach_phase_complete.float(),
        }
        self._previous_palm_distance = palm
        self._previous_finger_distance = finger
        self._previous_height = height.clone()
        self._maximum_height = updated_maximum_height
        self._previous_contacts = contacts
        self._previous_final_action = final_action.clone()
        self._previous_normalized_residual = effective_residual.clone()
        return reward, done, terms

    @torch.no_grad()
    def step(self, policy_action, step_id):
        if policy_action.shape[-1] != self.policy_action_dim:
            raise ValueError(
                "Expected policy action dimension {}, got {}".format(
                    self.policy_action_dim, policy_action.shape[-1]))
        normalized_residual = policy_action[:, :self.action_dim].clamp(-1.0, 1.0)
        if self.gated_residual:
            gate_action = policy_action[:, self.action_dim:].clamp(-1.0, 1.0)
            gates = 0.5 * (gate_action + 1.0)
            expanded_gate = torch.cat((
                gates[:, 0:1].expand(-1, 6),
                gates[:, 1:2].expand(-1, 22)), dim=-1)
            effective_residual = normalized_residual * expanded_gate
        else:
            gates = torch.ones(
                (self.num_envs, 2), device=self.device,
                dtype=normalized_residual.dtype)
            effective_residual = normalized_residual
        scaled_residual = effective_residual * self.residual_scale
        final_action = (self._bc_action + scaled_residual).clamp(-1.0, 1.0)
        self.task.step(final_action, int(step_id))
        reward, done, terms = self._reward_and_done(
            effective_residual, final_action, gates)
        processed = self._processed_observation()
        bc_action = self._predict_bc_action(processed)

        prop = processed[:, :self.prop_dim]
        if self.history_frames > 1:
            self._prop_history = torch.cat(
                (self._prop_history[:, 1:], prop[:, None, :]), dim=1)
            self._action_history = torch.cat((
                self._action_history[:, 1:], final_action[:, None, :]), dim=1)
        else:
            self._prop_history[:, 0] = prop
        self._bc_action = bc_action
        self._assemble_observations(processed, bc_action)
        return self._actor_obs, self._critic_obs, reward, done, terms
