import gym
import numpy as np

class ActionNoise:
    """
    Base class for action noise. It provides a method to get an action with optional noise.
    """
    def __init__(self, action_space: gym.spaces.Box = None):
        self.action_space = action_space

    def get_action(self, action):
        """
        Return the action with optional noise.
        """
        return action

class ContinuousActionNoise(ActionNoise):
    """
    Class to add continuous noise to actions in a given action space.
    """
    def __init__(self, action_space: gym.spaces.Box = None, mu: float = 0, sigma: float = 0.1):
        super().__init__(action_space)
        self.mu = mu
        self.sigma = sigma
        self.action_dim = self.action_space.shape[0]
        self.action_high = self.action_space.high
        self.action_low = self.action_space.low

    def add_noise(self, action):
        """
        Add Gaussian noise to the action and clip the result to the range [-1, 1].
        """
        noise = np.random.normal(self.mu, self.sigma, size=self.action_dim)
        noisy_action = action + noise
        return np.clip(noisy_action, -1, 1)

    def normalize_action(self, action):
        """
        Normalize the action to the range [-1, 1].
        """
        return 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1

    def denormalize_action(self, norm_action):
        """
        Denormalize the action from the range [-1, 1] to the original action space range.
        """
        return self.action_low + (norm_action + 1) * (self.action_high - self.action_low) / 2

    def get_action(self, action):
        """
        Get the action with added noise. The action is first normalized, noise is added,
        and then the action is denormalized back to the original action space range.
        """
        norm_action = self.normalize_action(action)
        noise_action = self.add_noise(norm_action)
        return self.denormalize_action(noise_action)
