import torch as th
import numpy as np
import torch.nn as nn
from ActionNoise import *

class BaseAgent:
    """
    BaseAgent:
    A basic reinforcement learning agent that uses a policy network to determine actions.
    """
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def act(self, obs):
        """
        Get the action from the policy network based on the observation.
        """
        pass

    def update_policy_net(self, policy_params):
        """
        Update the policy network's parameters.
        """
        self.policy_net.load_state_dict(policy_params)
        
class SB3Agent:
    """
    BaseAgent:
    A basic reinforcement learning agent that uses a policy network to determine actions.
    """
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def act(self, obs):
        """
        Get the action from the policy network based on the observation.
        """
        return self.policy_net.predict(obs)[0]

    def update_policy_net(self, policy_net):
        """
        Update the policy network's parameters.
        """
        self.policy_net = policy_net

class ContinuousActionAgent(SB3Agent):
    """
    ContinuousActionAgent:
    A reinforcement learning agent that adds noise to the actions in a continuous action space.
    """
    def __init__(self, policy_net, action_noise: ActionNoise):
        super().__init__(policy_net)
        self.action_noise = action_noise

    def act(self, obs):
        """
        Get the action from the policy network, add noise to it, and return the noisy action.
        """
        action = self.policy_net(obs)
        action = self.action_noise.get_action(action)
        return action
      
# class DiscreteActionAgent(BaseAgent):
