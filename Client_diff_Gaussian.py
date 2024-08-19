import torch
import torch as th
import numpy as np
import torch.nn as nn
import pandas as pd
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
from PredictionModel import PredictionModel, GaussianModel
from Utils import *
from sklearn.model_selection import train_test_split
import gymnasium as gym
from Agent import BaseAgent
import copy


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
        self.model = GaussianModel(self.obs_size, self.action_size, hidden_size).to(device)
        self.agent = agent
        self.policy = self.agent.policy_net

        self.lr = lr
        self.device = device
        
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
        train_X, test_X, train_y, test_y = train_test_split(self.dataset_X, self.dataset_y, test_size=0.3)

        train_X = expand_action_column(train_X, action_column_name="actions").to_numpy()
        train_y = train_y.to_numpy()
        test_X = expand_action_column(test_X, action_column_name="actions").to_numpy()
        test_y = test_y.to_numpy()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        overfit = 0
        last_test_avg = float('inf')

        for epoch in range(num_epoch):
            train_loss = self.model_train(train_X, train_y, optimizer, batch_size)
            test_avg = self.model_test(test_X, test_y, batch_size)

            print(f"Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_avg:.4f}")

            if test_avg > last_test_avg:
                overfit += 1
            else:
                overfit = 0
                last_test_avg = test_avg

            if overfit >= 10:
                print("Early stopping due to overfitting.")
                break

    # gymnasium
    def sample_data(self, n):
        '''
        Sample data using policy π to train the prediction model
        '''
        env = self.env
        observation, info = env.reset()
        df = pd.DataFrame(observation).T
        actions = []

        # Sample data from the environment
        for i in range(n):
            action = self.agent.act(observation)

            if i != 0:
                df.loc[len(df)] = observation
            actions.append(action)
            observation, reward, done, truncated, info = env.step(action)

            if done or truncated:
                observation, info = env.reset()
        env.close()

        df["actions"] = actions
        self.trajetories = df
        self.dataset_X, self.dataset_y = process_dfs_diff(df)
        
    def learn(self, timesteps = 1000, epoch = 10, batch_size = 32):
      self.sample_data(timesteps)
      
      self.train_prediction_model(timesteps, num_epoch = epoch, batch_size=batch_size)
      

    def model_train(self, X, y, optimizer, batch_size):
        self.model.train()
        loss_sum = 0

        for i in range(0, len(y), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
            batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.float32).to(self.device)

            optimizer.zero_grad()
            mean, logvar = self.model(batch_X)
            loss = self.gaussian_nll_loss(mean, logvar, batch_y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        return loss_sum / len(y)

    def model_test(self, X, y, batch_size):
        self.model.eval()
        loss_sum = 0

        with torch.no_grad():
            for i in range(0, len(y), batch_size):
                batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
                batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.float32).to(self.device)

                mean, logvar = self.model(batch_X)
                loss = self.gaussian_nll_loss(mean, logvar, batch_y)
                loss_sum += loss.item()

        return loss_sum / len(y)

    def gaussian_nll_loss(self, mean, logvar, target):
        inv_var = torch.exp(-logvar)
        loss = (mean - target) ** 2 * inv_var + logvar
        return loss.mean()
