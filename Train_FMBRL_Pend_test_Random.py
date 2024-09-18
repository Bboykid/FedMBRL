from sb3_contrib import TRPO
from Client_diff_RandomSample import FRLClient, model_test
from Agent import SB3Agent
import copy
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import numpy as np
import torch.nn as nn


from H_Envs.pendulum import PendulumEnv
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from MBEnvs.mb_pendulum_base import MB_PendulumEnv
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
    env_models = []
    MB_env = TimeLimit(MB_PendulumEnv(env_models,device), max_episode_steps = 200)
    
    # Global_RL = PPO("MlpPolicy", MB_env, verbose=1)
    Global_RL = TRPO("MlpPolicy", MB_env, verbose=1)
    
    train_loss_records = []
    test_loss_records = []
    
    # env_theta = [0.1, 0.3, 0.5, 0.7, 0.9]
    real_envs = []
    Clients = []
    for i in range(CLIENTS_NUM):
        real_envs.append( TimeLimit(PendulumEnv(g=env_paras[i]), max_episode_steps=200) )
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
    
        MB_env = TimeLimit(MB_PendulumEnv(env_models,device), max_episode_steps = 200)
        
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
        

if __name__ == '__main__':
    # initialize the client and server
	timesteps_real_per_round = 600
	timesteps_fc_per_round = timesteps_real_per_round * 30
	epoch_per_round = 10
	rounds_num = 50
	batch_size_env_model = 128
	test_dir= "H_Env_Base_Pen_G=10_RandomSample"
	env_paras = [10.0, 10.0, 10.0]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	exper_num = 3
	for exper_idx in range(exper_num):
		rewards_log, train_loss_log, test_loss_log = train(env_paras, device, test_dir, timesteps_real_per_round, timesteps_fc_per_round, epoch_per_round, rounds_num, batch_size_env_model)
		np.save( test_dir + "/" + str(exper_idx) + "_reward_logs.npy", rewards_log)
		np.save( test_dir + "/" + str(exper_idx) + "_train_loss_logs.npy", train_loss_log)
		np.save( test_dir + "/" + str(exper_idx) + "_test_loss_logs.npy", test_loss_log)