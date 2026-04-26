import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from app.rl.network import DQNNetwork


class DQNAgent:
    def __init__(self, state_size=9, action_size=3):
        """
        state_size:
        5 prices + price_change + moving_average + normalized_balance + position = 9

        action_size:
        0 = hold
        1 = buy
        2 = sell
        """

        self.state_size = state_size
        self.action_size = action_size

        self.network = DQNNetwork(state_size, action_size)

        self.target_network = copy.deepcopy(self.network)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95

    def preprocess_state(self, state):
        """
        Convert environment state into a flat numeric tensor.

        Final input:
        [
            normalized prices...,
            normalized price_change,
            normalized moving_avg,
            normalized balance,
            position
        ]
        """

        prices = state["prices"]
        balance = state["balance"]
        position = state["position"]

        first_price = prices[0]

        normalized_prices = [
            (price - first_price) / first_price
            for price in prices
        ]

        price_change = state.get("price_change", prices[-1] - prices[-2])
        normalized_price_change = price_change / first_price

        moving_avg = state.get("moving_avg", sum(prices) / len(prices))
        normalized_moving_avg = moving_avg / first_price

        normalized_balance = balance / 10000

        state_list = normalized_prices + [
            normalized_price_change,
            normalized_moving_avg,
            normalized_balance,
            position,
        ]

        return torch.tensor([state_list], dtype=torch.float32)

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy.
        """

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

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
                next_q_values = self.target_network(next_state_tensor)
                max_next_q = torch.max(next_q_values)

            target = reward

            if not done:
                target += self.gamma * max_next_q.item()

            target_q_values = q_values.clone().detach()
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
        Reduce randomness after each episode.
        """

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def update_target_network(self):
        """
        Copy main network weights into target network.
        """

        self.target_network.load_state_dict(self.network.state_dict())