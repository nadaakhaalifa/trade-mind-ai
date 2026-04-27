import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim

from app.rl.network import DQNNetwork


class DQNAgent:
    def __init__(self, state_size=10, action_size=3):
        """
        Actions:
        0 = hold
        1 = buy
        2 = sell
        """

        self.state_size = state_size
        self.action_size = action_size

        # Networks
        self.network = DQNNetwork(state_size, action_size)
        self.target_network = copy.deepcopy(self.network)
        self.target_network.eval()

        # Training
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

        # RL Params
        self.gamma = 0.99

        # Exploration (FIXED)
        self.epsilon = 1.0
        self.epsilon_decay = 0.98  
        self.epsilon_min = 0.05

    def preprocess_state(self, state):
        """
        Normalize environment state
        """

        prices = state["prices"]
        balance = state["balance"]
        position = state["position"]

        first_price = prices[0]

        normalized_prices = [(p - first_price) / first_price for p in prices]

        price_change = prices[-1] - prices[-2]
        normalized_price_change = price_change / first_price

        moving_avg = sum(prices) / len(prices)
        trend = prices[-1] - moving_avg
        normalized_trend = trend / first_price

        momentum = prices[-1] - prices[0]
        normalized_momentum = momentum / first_price

        normalized_balance = balance / 10000

        state_list = normalized_prices + [
            normalized_price_change,
            normalized_trend,
            normalized_balance,
            normalized_momentum,
            position,
        ]

        return torch.tensor([state_list], dtype=torch.float32)

    def choose_action(self, state):
        """
        Epsilon-greedy with FIXED exploration bias
        """

        # encourage buy/sell instead of HOLD spam
        if random.random() < self.epsilon:
             if state["position"] == 1:
                 return random.choice([0, 2])  # hold or sell only
             
             return random.choice([0, 1])      # hold or buy only
             
            
        state_tensor = self.preprocess_state(state)

        with torch.no_grad():
            q_values = self.network(state_tensor)

        return torch.argmax(q_values).item()

    def learn(self, batch):
        """
        Learn from batch (FIXED stability)
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

        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

        self.optimizer.step()

    def decay_epsilon(self):
        """
        Reduce randomness
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.epsilon = max(self.epsilon, self.epsilon_min)

    def update_target_network(self):
        """
        Sync networks
        """
        self.target_network.load_state_dict(self.network.state_dict())