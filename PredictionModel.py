import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from Config import HIDDEN_SIZE

class PredictionModel(nn.Module):
  def __init__(self, obs_size, action_size, hidden_size = HIDDEN_SIZE):
    super().__init__()
    self.net = nn.Sequential (
        nn.Linear(obs_size + action_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, obs_size),
    )
    
  def forward(self, x):
      return self.net(x)

class PredictionModel(nn.Module):
  def __init__(self, obs_size, action_size, hidden_size = HIDDEN_SIZE):
    super().__init__()
    self.net = nn.Sequential (
        nn.Linear(obs_size + action_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, obs_size),
    )
    
  def forward(self, x):
      return self.net(x)


class GaussianModel(nn.Module):
  def __init__(self, obs_size, action_size, hidden_size=HIDDEN_SIZE, learn_logvar_bounds=False):
    super(GaussianModel, self).__init__()
    self.out_size = obs_size
    
    # Define the fully connected network
    self.fc_net = nn.Sequential(
        nn.Linear(obs_size + action_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )
    
    # Linear layer to output mean and log variance
    self.mean_logvar = nn.Linear(hidden_size, obs_size * 2)
    
    # Learnable parameters for log variance bounds
    self.min_logvar = nn.Parameter(
      -10 * torch.ones(1, obs_size), requires_grad=learn_logvar_bounds
    )
    self.max_logvar = nn.Parameter(
      0.5 * torch.ones(1, obs_size), requires_grad=learn_logvar_bounds
    )
    
  def forward(self, x):
    # Pass input through the fully connected network
    x = self.fc_net(x)
    
    # Get mean and log variance
    mean_and_logvar = self.mean_logvar(x)
    mean = mean_and_logvar[..., :self.out_size]
    logvar = mean_and_logvar[..., self.out_size :]
    
    # Apply bounds to log variance using softplus function
    logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
    logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
    
    if logvar.shape[0] == 1:
            mean = mean.squeeze(0)
            logvar = logvar.squeeze(0)
    
    return mean, logvar
  
  
class DropoutMLP(nn.Module):
  def __init__(self, obs_size, action_size, hidden_size=HIDDEN_SIZE, dropout_prob=0.5):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(obs_size + action_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(p=dropout_prob),  # Add Dropout layer here
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(p=dropout_prob),  # Add Dropout layer here
      nn.Linear(hidden_size, obs_size),
      )

  def forward(self, x):
    return self.net(x)