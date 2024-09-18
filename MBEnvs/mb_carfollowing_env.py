__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np
import copy
import gym
import torch

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

'''
    model-based environment 构建注意事项
        1. ensemble models: 随机选择
        2. device设置统一
        3. space范围clip
        4. reward 返回值设置
        5. terminal函数设定
        
    此环境(MB_PendulumEnv)通过sb3的env_checker
    训练时需要观察models的更新。
'''


class MB_CarFollowing(gym.Env):

    def __init__(self, real_env, models, device):
        
		
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.action_space = copy.deepcopy(real_env.action_space)
        self.observation_space = copy.deepcopy(real_env.observation_space)
        '''
        	env models
        '''
        self.models = models
        self.device = device
        self.real_env = real_env
        
        self.TTC_threshold = real_env.TTC_threshold
        self.PF = real_env.config['policy_frequency']
        self.acc_th = real_env.acc_th
        self.acc_max = real_env.acc_max

    def step(self, u):
        
        self.cur_model = self.models[np.random.randint(0, len(self.models))]
        last_obs = self.obs  # th := theta
        
        reward = self._get_reward(self.obs, u)
        
        next_obs = self._get_obs_by_model(last_obs, action=u)
        self.obs = next_obs
        self.time_step_cnt += 1
        terminal = self._is_terminal(next_obs)

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return next_obs, reward, terminal, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs = self.real_env.reset()
        self.last_acceleration = obs[4]
        self.time_step_cnt = 0
        self.obs = obs
        return self.real_env.reset()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    
    def _get_obs_by_model(self, obs, action):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        input_d = torch.cat((obs_tensor, action_tensor))
        input_d = input_d.to(self.device)
        output_tensor = self.cur_model.forward(input_d)
        output_np = output_tensor.detach().cpu().numpy()
        
        obs = self.obs + output_np
        
        # Clip the output to the observation space range
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs_clipped
        
    def _get_reward(self, obs, action):
        front_space, ego_speed, front_speed, related_speed, ego_acc = obs
        
        acceleration = ego_acc
        jerk = (acceleration - self.last_acceleration) * self.PF
        self.last_acceleration = acceleration
        self.jerk = jerk
        if related_speed == 0:
            related_speed = 0.0001
        
        self.TTC = -front_space/related_speed

        if self.TTC >= 0 and self.TTC <= self.TTC_threshold:
            fTTC = np.log(self.TTC/self.TTC_threshold)
        else:
            fTTC = 0

        fSafety = fTTC

        mu = 0.422618
        sigma = 0.43659
        hdw = front_space/ego_speed
        if hdw <= 0:
            fHdw = -1
        else:
            fHdw = (np.exp(-(np.log(hdw) - mu) ** 2 / (2 * sigma ** 2)
                           ) / (hdw * sigma * np.sqrt(2 * np.pi)))
        
        fEffic = fHdw
        
        acc_abs = abs(acceleration)
        jerk_abs = abs(jerk)
        acc_abs = min(acc_abs,4)
        jerk_abs = min(jerk_abs,60)
        if acc_abs > self.acc_th:
            fAcc = - ((acc_abs - self.acc_th)/(self.acc_max - self.acc_th))**2
        else:
            fAcc = 0
        
        # fJerk = -(jerk_abs/60)**2
        fJerk = -((jerk**2)/ (self.PF * 6)**2)
        fJerk = float(fJerk)
        fAcc = float(fAcc)
        fComf = fJerk
        
        self.fJerk = float(fJerk)
        self.fAcc = float(fAcc)
        
        if front_space < 0:
            fCrash = -100
        else:
            fCrash = 0
            
        totalReward = fSafety + fComf + fEffic + fCrash

        self.total_reward = float(totalReward)
        self.total_risk = float(fSafety)
        self.front_risk = float(fSafety)
        self.fRisk = float(fSafety)
        # self.front_left_collision_risk = (f_left_weight, f_left_risk)
        # self.rear_left_collision_risk = (r_left_weight, r_left_risk)
        # self.front_right_collision_risk = (f_right_weight, f_right_risk)
        # self.rear_right_collision_risk = (r_right_weight, r_right_risk)
        self.fComf = float(fComf)
        self.fEffic = float(fEffic)

        return float(totalReward)
        
    
    def _is_terminal(self, obs):
        if obs[0] <= 0 or self.time_step_cnt > 200:
            return False
        else:
            return True

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            
    def _recover_theta(self,obs1, obs2):
        theta = np.arctan2(obs2, obs1)
        return theta


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi