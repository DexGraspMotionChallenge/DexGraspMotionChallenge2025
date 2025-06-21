import gym
from gym import spaces
import numpy as np
import torch
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n, mode='stack'):
    x = list(x)
    n = min(len(x), n)
    
    if isinstance(x[0], torch.Tensor):
        if mode=='stack':
            return torch.stack(x[-n:])
        elif mode=='max':
            return torch.stack(x[-n:]).max(dim=0)[0]

    else:
        return np.array(x[-n:])




def dict_take_last_n(x, n, mode='stack'):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n, mode)
    return result


def aggregate(data, method='max'):
    if isinstance(data[0], torch.Tensor):
        if method == 'max':
            # equivalent to any
            return torch.max(torch.stack(data),dim=0)[0]
        elif method == 'min':
            # equivalent to all
            return torch.min(torch.stack(data),dim=0)[0]
        elif method == 'mean':
            return torch.mean(torch.stack(data),dim=0)
        elif method == 'sum':
            return torch.sum(torch.stack(data),dim=0)
        else:
            raise NotImplementedError()
    else:
        if method == 'max':
            # equivalent to any
            return np.max(data)
        elif method == 'min':
            # equivalent to all
            return np.min(data)
        elif method == 'mean':
            return np.mean(data)
        elif method == 'sum':
            return np.sum(data)
        else:
            raise NotImplementedError()



def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype).to(all_obs[-1].device)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='sum'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.previous_acts = deque(maxlen=3)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        # self.reward = list()
        # self.done = list()

        self.reward = deque(maxlen=self.n_action_steps)
        self.done = deque(maxlen=self.n_action_steps)

        # self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))
        self.info = defaultdict(lambda : deque(maxlen=self.n_action_steps))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action, id=0):
        """
        actions: (n_action_steps,) + action_shape
        """
        for i,act in enumerate(action):
            # if len(self.done) > 0 and self.done[-1]:
                # termination
                # break
            # observation, reward, done, info = self.env.step(act, id+i)
            if id+i>33 and self.env.task.cfg['env']['use_pre_fixed_actions']: #40
                act = self.env.task.get_pre_target_actions(act,id+i)
                # act = self.env.task.get_pre_target_actions(act,id+i,previous_actions=self.previous_acts[0])

            self.env.task.step(act, id+i)
            a=1

            # self.obs.append(observation.clone())
            # self.reward.append(reward)
            self.obs.append(self.env.task.obs_buf.clone())
            self.previous_acts.append(act.clone())

        #     if (self.max_episode_steps is not None) \
        #         and (len(self.reward) >= self.max_episode_steps):
        #         # truncation
        #         done = True
        #     self.done.append(done)
        #     self._add_info(info)
        #
        observation = self._get_obs(self.n_obs_steps)
        # reward = aggregate(self.reward, self.reward_agg_method)
        # done = aggregate(self.done, 'max')
        # info = dict_take_last_n(self.info, self.n_obs_steps)
        # return observation, reward, done, info
        return observation, 0, 0, 0

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
