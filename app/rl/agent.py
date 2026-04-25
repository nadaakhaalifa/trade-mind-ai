import random
import torch

from app.rl.network import DQNNetwork


class DQNAgent:
    def __init__(self, state_size=7, action_size=3):
        """
        state_size: number of inputs (prices + balance + position)
        action_size: number of actions (hold, buy, sell)
        """
        self.state_size = state_size
        self.action_size = action_size

        # neural network (brain)
        self.network = DQNNetwork(state_size, action_size)

        # exploration rate
        self.epsilon = 1.0  # start fully random

    def preprocess_state(self, state):
        """
        Convert state dict → flat tensor
        """
        prices = state["prices"]
        balance = state["balance"]
        position = state["position"]

        state_list = prices + [balance, position]

        return torch.tensor([state_list], dtype=torch.float32)

    def choose_action(self, state):
        """
        Epsilon-greedy action selection
        """
        # random action
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # use neural network
        state_tensor = self.preprocess_state(state)

        with torch.no_grad():
            q_values = self.network(state_tensor)

        action = torch.argmax(q_values).item()

        return action