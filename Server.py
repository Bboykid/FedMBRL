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
from Client import model_train, model_test

class FRLServer():
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

        self.env_models = []

        self.lr = lr
        self.device = device

    def set_env(self, mb_env):
        self.env = mb_env

    def get_dis_model(self):
        return copy.deepcopy(self.model)

    def get_dis_model_params(self):
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
        train_X, test_X, train_y, test_y = train_test_split(self.dataset_X, self.dataset_y, test_size=0.1)

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

    def learn_dis_model(self, timesteps = 1000, epoch = 10, batch_size = 32):
      self.sample_data(timesteps)
      self.train_prediction_model(timesteps, num_epoch = epoch, batch_size=batch_size)

