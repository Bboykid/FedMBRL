import torch
import torch as th
import numpy as np
import torch.nn as nn
import pandas as pd
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
from PredictionModel import PredictionModel
from Utils import *
from sklearn.model_selection import train_test_split
import gymnasium as gym
from Agent import BaseAgent
import copy

DATA_MAX = 10000

class FRLClient():
    '''
      FRLClient:
      Train predcition model(predict the state trainsition) by sampled data
      data is sampled by the true environment 
      
      env: true environment
      model: prediction the transition(train the model by supervised learning)
      agent: interactive with the true environment by policy pi with explorating
	  
      params:
	    lr: learning_rate of model
	    hidden_size:
	    device:
    '''
  
    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        lr: float = 3e-4,
        hidden_size: int = 256,
        device: Union[th.device, str] = "auto",
      ):
        # Initialize true environment
        self.env = env
        self.obs_size = env.observation_space.shape[0]
        # Check if action_space is Discrete
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_size = 1
        else:
            self.action_size = env.action_space.shape[0]
        self.hidden_size = hidden_size
        
        # Initialize prediction model (predict the state transition)
        self.model = PredictionModel(self.obs_size, self.action_size, hidden_size).to(device)
        self.agent = agent
        self.policy = self.agent.policy_net

        self.lr = lr
        self.device = device
        self.dataset_X = None
        self.dataset_y = None
        self.dataMax = DATA_MAX
        
    def get_prediction_model(self):
        # Return the prediction model
        return copy.deepcopy(self.model)

    def get_prediction_model_params(self):
        # Return a deepcopy of the model's state dictionary
        return copy.deepcopy(self.model.state_dict())
      
    def update_policy(self, policy_net):
        # update the policy pi by policy parameters sended by server
        self.agent.update_policy_net(policy_net)
        self.policy = self.agent.policy_net

    def train_prediction_model(self, num_data=1000, num_epoch=100, batch_size=32):
        '''
        Train the prediction model with true data
        '''
        self.train_loss_list = []
        self.test_avg_list = []
        # Split the dataset
        train_X, test_X, train_y, test_y = train_test_split(self.dataset_X, self.dataset_y, test_size=0.2)
        
        # Flatten the column "actions"
        train_X = expand_action_column(train_X, action_column_name="actions")
        test_X = expand_action_column(test_X, action_column_name="actions")
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y

        model = self.model.to(self.device)
        
        # Define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        overfit = 0
        last_test_avg = 10000

        # Convert data to numpy arrays
        train_X = train_X.to_numpy()
        train_y = train_y.to_numpy()
        test_X = test_X.to_numpy()
        test_y = test_y.to_numpy()

        # Train the model for a given number of epochs
        for t in range(num_epoch):
            train_loss = model_train(train_X, train_y, self.model, loss_fn, optimizer, batch_size)
            test_avg = model_test(test_X, test_y, self.model, loss_fn, batch_size)
            self.train_loss_list.append(train_loss)
            self.test_avg_list.append(test_avg)
            # Early stopping if overfitting occurs
            if test_avg > last_test_avg:
                overfit += 1
            else:
                overfit = 0
                last_test_avg = test_avg
            if overfit >= 10:
                break
        
        self.model = model
        
    def sample_seq_data(self, n, seq_length):
        env = self.env
        
        # observation, info = env.reset()
        # df = pd.DataFrame(observation).T
        # actions = []
        dataset_X_list = []
        dataset_y_list = []
        cnt = 0
        while(cnt < n):
            observation, info = env.reset()
            obs_df = pd.DataFrame(observation).T 
            actions = []
            done = False
            truncated = False
            while done == False and truncated == False:
                action = self.agent.act(observation)
                obs_df[len(df_temp)] = observation
                actions.append(action)
                observation, reward, done, truncated, info = env.step(action)
                cnt += 1
                if cnt >= n:
                    break
            # df_temp["actions"] = actions
            dataset_X_temp, dataset_y_temp = create_sequences_diff(obs_df, action_df, seq_length)
            dataset_X_list.append(dataset_X_temp)
            dataset_y_list.append(dataset_y_temp)
        
            # 展平数据集
        dataset_X = np.concatenate(dataset_X_list, axis=0)
        dataset_y = np.concatenate(dataset_y_list, axis=0)
        self.cur_dataset_X, self.cur_dataset_y = dataset_X, dataset_y
            
        

    # # gymnasium
    # def sample_data(self, n):
    #     '''
    #     Sample data using policy π to train the prediction model
    #     '''
    #     env = self.env
    #     observation, info = env.reset()
    #     df = pd.DataFrame(observation).T
    #     actions = []

    #     # Sample data from the environment
    #     for i in range(int(n * 0.5)):
    #         action = self.agent.act(observation)

    #         if i != 0:
    #             df.loc[len(df)] = observation
    #         actions.append(action)
    #         observation, reward, done, truncated, info = env.step(action)

    #         if done or truncated:
    #             observation, info = env.reset()
    #     # env.close()
        
    #     observation, info = env.reset()
    #     for i in range(int(n * 0.5)):
    #         # action = self.agent.act(observation)
    #         action = env.action_space.sample()
    #         # if i != 0:
    #         df.loc[len(df)] = observation
    #         actions.append(action)
    #         observation, reward, done, truncated, info = env.step(action)

    #         if done or truncated:
    #             observation, info = env.reset()
    #     env.close()

    #     df["actions"] = actions
    #     self.trajetories = df
    #     self.cur_dataset_X, self.cur_dataset_y = process_dfs_diff(df)
    #     self.addData()
        
    def addData(self):
        """
        Add cur_dataset_X and cur_dataset_y to the main dataset_X and dataset_y.
        If the combined dataset exceeds the maximum size (self.dataMax), 
        randomly delete a batch of data to make space for the new data.
        """
        if self.dataset_X is None or self.dataset_y is None:
            # If the main dataset is not initialized, initialize it with the current data
            self.dataset_X = self.cur_dataset_X.copy()
            self.dataset_y = self.cur_dataset_y.copy()
        else:
            # Check if the combined dataset exceeds the maximum size
            if len(self.dataset_X) + len(self.cur_dataset_X) > self.dataMax:
                # Calculate the number of rows to remove
                excess_size = (len(self.dataset_X) + len(self.cur_dataset_X)) - self.dataMax
    
                # Randomly select indices to remove
                drop_indices = np.random.choice(self.dataset_X.index, excess_size, replace=False)
    
                # Drop rows from both dataset_X and dataset_y
                self.dataset_X = self.dataset_X.drop(drop_indices).reset_index(drop=True)
                self.dataset_y = self.dataset_y.drop(drop_indices).reset_index(drop=True)
    
            # Append new data
            self.dataset_X = pd.concat([self.dataset_X, self.cur_dataset_X], ignore_index=True)
            self.dataset_y = pd.concat([self.dataset_y, self.cur_dataset_y], ignore_index=True)
        
    def learn(self, timesteps = 1000, epoch = 10, batch_size = 32):
      self.sample_data(timesteps)
      
      self.train_prediction_model(timesteps, num_epoch = epoch, batch_size=batch_size)
      
    def get_dataset(self):
        return copy.deepcopy(self.train_X), copy.deepcopy(self.train_y),  copy.deepcopy(self.test_X),  copy.deepcopy(self.test_y)
      


def model_train(X, y, model, loss_fn, optimizer, batch_size):
    # Set the model to training mode
    model.train()
    loss_sum = 0

    # Loop through the dataset in batches
    for i in range(round((len(y) / batch_size) + 0)):
        # Convert batch data to torch tensors and move to GPU
        train_X = torch.from_numpy(X[i * batch_size: (i+1) * batch_size]).cuda()
        train_y = torch.from_numpy(y[i * batch_size: (i+1) * batch_size]).cuda()

        loss = 0

        # Compute loss for each sample in the batch
        for k in range(min(batch_size, len(train_X))):
            pred = model.forward(train_X[k].float())
            loss += loss_fn(pred.to(torch.float32), train_y[k].to(torch.float32))
        loss_sum += loss.item()

        # Zero the gradients, perform backpropagation, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Return average loss per sample
    return loss_sum / len(y)

def model_test(X, y, model, loss_fn, batch_size):
    loss_sum = 0
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        # Loop through the dataset in batches
        for i in range(round((len(y) / batch_size) + 1)):
            # Convert batch data to torch tensors and move to GPU
            test_X = torch.from_numpy(X[i * batch_size: (i+1) * batch_size]).cuda()
            test_y = torch.from_numpy(y[i * batch_size: (i+1) * batch_size]).cuda()

            # Compute loss for each sample in the batch
            for k in range(min(batch_size, len(test_X))):
                pred = model.forward(test_X[k].float())
                
                # loss += loss_fn(pred.to(torch.float32), test_y[k].to(torch.float32))
                loss = loss_fn(pred, test_y[k].float())
                loss_sum += loss.item()
                

    # Compute and print average loss per sample
    loss_sum /= len(y)
    print(f"Avg loss: {loss_sum}!")
    return loss_sum

def create_sequences(data, sequence_length):
    """ 将时间序列数据划分为样本，每个样本长度为 sequence_length """
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        # 获取长度为 sequence_length 的序列
        seq = data[i:i+sequence_length]
        # 目标是下一个时间步的数据
        target = data[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def create_sequences_diff(obs_df, action_df, sequence_length):
    """
    生成时间序列样本，并将 action 作为 X 的一部分。
    Y 是 observation 的差值 (next_obs - current_obs)。
    
    obs_df: 包含 observation 的 DataFrame
    action_df: 包含 action 的 DataFrame
    sequence_length: 序列的长度
    """
    sequences = []
    targets = []
    
    # 计算整个 obs_df 的差分
    obs_diff_df = obs_df.diff().dropna().reset_index(drop=True)
    
    # 确保 obs_df 和 action_df 长度相同
    assert len(obs_df) == len(action_df), "Observation and Action dataframes must have the same length."
    
    for i in range(len(obs_diff_df) - sequence_length):  # 确保不会越界
        # 获取长度为 sequence_length 的 observation 和 action 序列
        obs_seq = obs_df.iloc[i:i+sequence_length].values
        action_seq = action_df.iloc[i:i+sequence_length].values
        
        # 将 observation 和 action 合并为 X，action 是 X 的一部分
        sequence = np.concatenate([obs_seq, action_seq], axis=1)
        
        # 获取目标 Y，使用预计算的 obs_diff_df 差分序列
        target = obs_diff_df.iloc[i:i+sequence_length].values
        
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)