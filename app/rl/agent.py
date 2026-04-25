import random
import torch
import torch.nn as nn
import torch.optim as optim

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

        # optimizer (updates the network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

        # loss function (how wrong the model is)
        self.criterion = nn.MSELoss()

        # discount factor (importance of future rewards)
        self.gamma = 0.99

        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def preprocess_state(self, state):
        """
        Convert state dict → flat tensor.
        """
        prices = state["prices"]
        balance = state["balance"]
        position = state["position"]

        state_list = prices + [balance, position]

        return torch.tensor([state_list], dtype=torch.float32)

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy.
        """
        # exploration: choose random action
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # exploitation: choose best action from network
        state_tensor = self.preprocess_state(state)

        with torch.no_grad():
            q_values = self.network(state_tensor)

        return torch.argmax(q_values).item()

    def learn(self, batch):
        """
        Learn from a batch of experiences.
        """
        states = []
        targets = []

        for state, action, reward, next_state, done in batch:
            state_tensor = self.preprocess_state(state)
            next_state_tensor = self.preprocess_state(next_state)

            q_values = self.network(state_tensor)

            with torch.no_grad():
                next_q_values = self.network(next_state_tensor)
                max_next_q = torch.max(next_q_values)

            target = reward

            if not done:
                target += self.gamma * max_next_q.item()

            target_q_values = q_values.clone()
            target_q_values[0][action] = target

            states.append(state_tensor)
            targets.append(target_q_values)

        states = torch.cat(states)
        targets = torch.cat(targets)

        predictions = self.network(states)

        loss = self.criterion(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """
        Reduce randomness slowly after each episode.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min