from sb3_contrib import TRPO
from Client_diff_Emb import FRLClient
from Agent import SB3Agent
import copy
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import numpy as np


from H_Envs.pendulum import PendulumEnv
from H_Envs.pendulum_emb import PendulumEnvEmb
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from MBEnvs.mb_pendulum_emb import MB_PendulumEnv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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
    embs = [np.array([x]) for x in env_paras]
    env_models = []
    MB_env = TimeLimit(MB_PendulumEnv(env_models, embs,device), max_episode_steps = 200)
    
    # Global_RL = PPO("MlpPolicy", MB_env, verbose=1)
    Global_RL = TRPO("MlpPolicy", MB_env, verbose=1)
    
    # env_theta = [0.1, 0.3, 0.5, 0.7, 0.9]
    real_envs = []
    eva_envs = []
    Clients = []
    for i in range(CLIENTS_NUM):
        real_envs.append( TimeLimit(PendulumEnv(g=env_paras[i]), max_episode_steps=200) )
        eva_envs.append( TimeLimit(PendulumEnvEmb(g=env_paras[i]), max_episode_steps=200) )
        policy_net = Global_RL
        agent = SB3Agent(policy_net)
        client = FRLClient(real_envs[i], agent, lr = 3e-4, hidden_size = 256, device = device, emb=embs[i])
        Clients.append(client)
        env_model = copy.deepcopy(client.model)
        env_models.append(env_model)
        
    
        
    Global_RL.env.models = env_models
    
    Global_RL.save(model_tmp_path)
    
    rewards_log = []
    
    env_models = []
    for round_idx in range(rounds_num):
        print('------------------------------')
        print("round: " + str(round_idx))
        env_models = []
        for client_idx in range(len(Clients)):
            print('------------------------------')
            print("client: " + str(client_idx))
            # update policy
            Clients[client_idx].agent.policy_net = Global_RL
            # train prediction models
            Clients[client_idx].learn(timesteps_real_per_round, epoch_per_round, batch_size_env_model)
            #
            env_model = Clients[client_idx].get_prediction_model()
            env_models.append(env_model)
        
    #     Server.update_env_models(env_models)
    
        MB_env = TimeLimit(MB_PendulumEnv(env_models,embs,device), max_episode_steps = 200)
        
        Global_RL = TRPO.load(model_tmp_path, env = MB_env)
    #     Global_RL.env.models = env_models
        #
        Global_RL.learn(total_timesteps=timesteps_fc_per_round)
        
        Global_RL.save(model_tmp_path)
    #     Server.learn(timesteps_real_per_round = 10000)
        # test performance
        
        round_reward = []
        
        for client_idx in range(CLIENTS_NUM):
            mean_reward, std_reward = evaluate_policy(Global_RL, eva_envs[client_idx], n_eval_episodes=20)
            round_reward.append(mean_reward)
        rewards_log.append(round_reward)
        print("mean_reward in real env:" + str(round_reward))
        
    return rewards_log
        

if __name__ == '__main__':
    # initialize the client and server
	timesteps_real_per_round = 1000
	timesteps_fc_per_round = timesteps_real_per_round * 30
	epoch_per_round = 10
	rounds_num = 50
	batch_size_env_model = 128
	test_dir= "H_Env_Pen_Emb_Env9_Test2_TS1000"
	env_paras = [7.0,7.0,7.0, 10.0,10.0,10.0, 13.0,13.0,13.0]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	exper_num = 5
	for exper_idx in range(exper_num):
		rewards_log =train(env_paras, device, test_dir, timesteps_real_per_round, timesteps_fc_per_round, epoch_per_round, rounds_num, batch_size_env_model)
		np.save( test_dir + "/" + str(exper_idx) + "_reward_logs.npy", rewards_log)