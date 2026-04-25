import torch
import torch.nn as nn # neural network

# PyTorch model
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        state_size: number of input values the network receives
        action_size: number of actions the network can choose
        """
        super().__init__() #PyTorch parent class

        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        """
        Runs the state through the network and returns action scores.
        """
        return self.model(state)