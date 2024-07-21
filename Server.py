import copy

# Base class for Federated Model-Based Reinforcement Learning (FMBRL) Server
class FMBRL_Server:
    def __init__(self) -> None:
        pass
    
    def learn(self, timesteps=1000):
        """
        Method for learning/updating the policy.
        To be implemented by subclasses.
        """
        pass

    def get_policy_net_params(self):
        """
        Method to get the parameters of the policy network.
        To be implemented by subclasses.
        """
        pass
    
    def update_env_models(self, env_models):
        """
        Method to update the environment models.
        To be implemented by subclasses.
        """
        pass

# Subclass of FMBRL_Server using Stable Baselines3 (SB3) for Model-Based RL
class SB3_FMBRL_Server(FMBRL_Server):
    
    def __init__(self, RL_Policy) -> None:
        """
        Initialize with a specific RL policy.
        
        Args:
            RL_Policy: The RL policy used for training and updates.
        """
        super().__init__()
        self.RL_Policy = RL_Policy
        
    def get_policy_net_params(self):
        """
        Get the parameters of the policy network.
        To be implemented by subclasses.
        """
        pass
    
    def update_env_models(self, env_models):
        """
        Update the environment models in the RL policy.
        
        Args:
            env_models: The new environment models to be updated.
        """
        self.RL_Policy.env.models = env_models
    
    def learn(self, timesteps=1000):
        """
        Learn/update the policy using the given timesteps.
        
        Args:
            timesteps: The number of timesteps for training.
        """
        self.RL_Policy.learn(timesteps)
        

class PPO_Server(SB3_FMBRL_Server):
    
    def __init__(self, RL_Policy) -> None:
        """
        Initialize with a specific PPO policy.
        
        Args:
            RL_Policy: The PPO policy used for training and updates.
        """
        super().__init__(RL_Policy)
    
    def get_policy_net_params(self):
        """
        Get the parameters of the PPO policy network.
        
        Returns:
            A deepcopy of the policy network parameters.
        """
        return copy.deepcopy(self.RL_Policy.policy.state_dict())
	
 
    


	