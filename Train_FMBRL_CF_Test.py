from sb3_contrib import TRPO
from Client_diff_new_NewSample import FRLClient, model_test
from Agent import SB3Agent
import copy
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import numpy as np
import torch.nn as nn


# from H_Envs.pendulum import PendulumEnv
import gymnasium
from gymnasium.wrappers import TimeLimit
from MBEnvs.mb_carfollowing_env import MB_CarFollowing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import gym
import gymnasium as gymnasium
import gymnasium.spaces as spaces
import numpy as np

class GymToGymnasiumWrapper(gymnasium.Env):
    def __init__(self, gym_env):
        super(GymToGymnasiumWrapper, self).__init__()
        self.gym_env = gym_env
        # 将 gym 的 space 转换为 gymnasium 的 space
        self.observation_space = self._convert_space(gym_env.observation_space)
        self.action_space = self._convert_space(gym_env.action_space)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.gym_env.seed(seed)
        observation = self.gym_env.reset()
        return observation, {}

    def step(self, action):
        observation, reward, done, info = self.gym_env.step(action)
        return observation, reward, done, False, info

    def render(self, mode="human"):
        return self.gym_env.render(mode=mode)

    def close(self):
        return self.gym_env.close()

    def seed(self, seed=None):
        return self.gym_env.seed(seed)

    def _convert_space(self, space):
        """将 gym 空间转换为 gymnasium 空间"""
        if isinstance(space, gym.spaces.Box):
            return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, gym.spaces.Discrete):
            return spaces.Discrete(n=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return spaces.MultiDiscrete(nvec=space.nvec)
        elif isinstance(space, gym.spaces.MultiBinary):
            return spaces.MultiBinary(n=space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return spaces.Tuple([self._convert_space(s) for s in space.spaces])
        elif isinstance(space, gym.spaces.Dict):
            return spaces.Dict({key: self._convert_space(s) for key, s in space.spaces.items()})
        else:
            raise NotImplementedError(f"空间类型 {type(space)} 还未实现转换")

import sys
sys.path.append("F:/AI/ACC_ENV")
from my_highway_env.envs.car_following_env_by_config_dynamical import CarFollowingEnv


# # initialize the client and server
# timesteps_real_per_round = 500
# timesteps_fc_per_round = timesteps_real_per_round * 30
# epoch_per_round = 100
# CLIENTS_NUM = 3
# rounds_num = 30
# batch_size_env_model = 128

# test_dir = "Diff_test2"
# model_tmp_path = test_dir + "/models/tmp"

def train(env_paras, device, save_dir, timesteps_real_per_round = 500, timesteps_fc_per_round = 20000, epoch_per_round = 100, rounds_num = 30, batch_size_env_model = 128):
    
    model_tmp_path = save_dir + "/models/tmp"
    
    CLIENTS_NUM = len(env_paras)
    
    test_env = CarFollowingEnv(env_paras[0])
    env_models = []
    MB_env = TimeLimit(GymToGymnasiumWrapper(MB_CarFollowing(test_env, env_models,device)), max_episode_steps = 100)
    
    # Global_RL = PPO("MlpPolicy", MB_env, verbose=1)
    Global_RL = TRPO("MlpPolicy", MB_env, verbose=1)
    
    train_loss_records = []
    test_loss_records = []
    
    # env_theta = [0.1, 0.3, 0.5, 0.7, 0.9]
    real_envs = []
    Clients = []
    for i in range(CLIENTS_NUM):
        real_envs.append( GymToGymnasiumWrapper(CarFollowingEnv(env_paras[i])) )
        policy_net = Global_RL
        agent = SB3Agent(policy_net)
        client = FRLClient(real_envs[i], agent, lr = 3e-4, hidden_size = 256, device = device)
        Clients.append(client)
        env_model = copy.deepcopy(client.model)
        env_models.append(env_model)
        
    
        
    Global_RL.env.models = env_models
    
    Global_RL.save(model_tmp_path)
    
    rewards_log = []
    
    loss_fn = nn.MSELoss()
    
    env_models = []
    for round_idx in range(rounds_num):
        print('------------------------------')
        env_models = []
        print("round: " + str(round_idx))
        
        # r_train_X = []
        # r_train_y = [] 
        # r_test_X =[]
        # r_test_y= []
        r_dataset = []
        
        cur_round_train_loss = []
        cur_round_test_loss = []
        for client_idx in range(len(Clients)):
            print('------------------------------')
            print("client: " + str(client_idx))
            # update policy
            Clients[client_idx].agent.policy_net = Global_RL
            # train prediction models
            Clients[client_idx].learn(timesteps_real_per_round, epoch_per_round, batch_size_env_model)
            
            c_train_X, c_train_y, c_test_X, c_test_y = Clients[client_idx].get_dataset()
            r_dataset.append((c_train_X, c_train_y, c_test_X, c_test_y))
            # r_train_X.append(c_train_X)
            # r_train_y.append(c_train_y)
            # r_test_X.append(c_test_X)
            # r_test_y.append(c_test_y)
            
            
            # cur_round_train_loss.append(train_loss)
            # cur_round_test_loss.append(test_loss)
            #
            env_model = Clients[client_idx].get_prediction_model()
            env_models.append(env_model)
        
        cur_round_train_loss = []
        cur_round_test_loss = []
        for eva_client_idx in range(len(Clients)):
            client_train_loss = []
            client_test_loss = []
            c_model = env_models[eva_client_idx]
            for dataset_idx in range(len(Clients)):
                c_train_X, c_train_y, c_test_X, c_test_y = r_dataset[dataset_idx]
                train_loss = model_test(c_train_X.to_numpy(), c_train_y.to_numpy(), c_model, loss_fn, batch_size_env_model)
                test_loss = model_test(c_test_X.to_numpy(), c_test_y.to_numpy(), c_model, loss_fn, batch_size_env_model)
                client_train_loss.append(train_loss)
                client_test_loss.append(test_loss)
            cur_round_train_loss.append(client_train_loss)
            cur_round_test_loss.append(client_test_loss)
        
        
        train_loss_records.append(cur_round_train_loss)
        test_loss_records.append(cur_round_test_loss)
        
    #     Server.update_env_models(env_models)
    
        MB_env = TimeLimit(GymToGymnasiumWrapper(MB_CarFollowing(test_env, env_models,device)), max_episode_steps = 100)

        
        Global_RL = TRPO.load(model_tmp_path, env = MB_env)
    #     Global_RL.env.models = env_models
        #
        Global_RL.learn(total_timesteps=timesteps_fc_per_round)
        
        Global_RL.save(model_tmp_path)
    #     Server.learn(timesteps_real_per_round = 10000)
        # test performance
        
        round_reward = []
        
        for client_idx in range(CLIENTS_NUM):
            mean_reward, std_reward = evaluate_policy(Global_RL, real_envs[client_idx], n_eval_episodes=20)
            round_reward.append(mean_reward)
        rewards_log.append(round_reward)
        print("mean_reward in real env:" + str(round_reward))
        
    return rewards_log, train_loss_records, test_loss_records
        

dataset_path_train = "D:/Dataset/data/CarFollowing/benchmark/" + "NGSIM_I_80" + "_" + "train" + "_data.npy"
# dataset_path_val = "/ai/syf/ACC/Dataset/ProcessedData/NGSIM_I_80_val_data_C.npy"
obs_type = 'BaselineAccObservation'
PF = 10

env_config_NGSIM_train_1 = {
    'observation': {"type": obs_type, 'noise': False},
    'action': {"type": "ContinuousAction", "acceleration_range": (-4, 2),"jerk_max":60,'lateral':False,"dynamical": False},
    # 'dataset_path': "D:/Dataset/data/CarFollowing/benchmark/" + "NGSIM_I_80" + "_" + "train" + "_data.npy",
    'dataset_path': dataset_path_train,
    'simulation_frequency':10,
    'policy_frequency': PF,
	'vehicle': {"tau": 0, "delay": 0},
}

env_config_NGSIM_train_2 = {
    'observation': {"type": obs_type, 'noise': False},
    'action': {"type": "ContinuousAction", "acceleration_range": (-4, 2),"jerk_max":60,'lateral':False,"dynamical": False},
     # 'dataset_path': "D:/Dataset/data/CarFollowing/benchmark/" + "NGSIM_I_80" + "_" + "train" + "_data.npy",
    'dataset_path': dataset_path_train,
    'simulation_frequency':10,
    'policy_frequency': PF,
	'vehicle': {"tau": 0.2, "delay": 0},
}

env_config_NGSIM_train_3 = {
    'observation': {"type": obs_type, 'noise': False},
    'action': {"type": "ContinuousAction", "acceleration_range": (-4, 2),"jerk_max":60,'lateral':False,"dynamical": False},
     # 'dataset_path': "D:/Dataset/data/CarFollowing/benchmark/" + "NGSIM_I_80" + "_" + "train" + "_data.npy",
    'dataset_path': dataset_path_train,
    'simulation_frequency':10,
    'policy_frequency': PF,
	'vehicle': {"tau": 0.4, "delay": 0},
}

if __name__ == '__main__':
    # initialize the client and server
    timesteps_real_per_round = 2000
    timesteps_fc_per_round = timesteps_real_per_round * 10
    epoch_per_round = 100
    rounds_num = 5
    batch_size_env_model = 128
    test_dir= "CF_ENV_Base_Pen_TS_1k_Log_NewSample"
    env_paras = [env_config_NGSIM_train_3,env_config_NGSIM_train_3,env_config_NGSIM_train_3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exper_num = 3
    for exper_idx in range(exper_num):
        rewards_log, train_loss_log, test_loss_log = train(env_paras, device, test_dir, timesteps_real_per_round, timesteps_fc_per_round, epoch_per_round, rounds_num, batch_size_env_model)
        np.save( test_dir + "/" + str(exper_idx) + "_reward_logs.npy", rewards_log)
        np.save( test_dir + "/" + str(exper_idx) + "_train_loss_logs.npy", train_loss_log)
        np.save( test_dir + "/" + str(exper_idx) + "_test_loss_logs.npy", test_loss_log)