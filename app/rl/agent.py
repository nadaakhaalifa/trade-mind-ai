import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim

from app.rl.network import DQNNetwork


class DQNAgent:
    def __init__(self, state_size=15, action_size=3):
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

        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.985
        self.epsilon_min = 0.05

    def preprocess_state(self, state):
        """
        Normalize environment state
        """

        prices = state["prices"]
        balance = state["balance"]
        position = state["position"]

        first_price = prices[0] + 1e-9

        normalized_prices = [
            (price - first_price) / first_price
            for price in prices
        ]

        price_change = state.get("price_change", prices[-1] - prices[-2])
        normalized_price_change = price_change / first_price

        momentum = state.get("momentum", prices[-1] - prices[0])
        normalized_momentum = momentum / first_price

        moving_avg = state.get("moving_avg", sum(prices) / len(prices))
        trend = state.get("trend", prices[-1] - moving_avg)
        normalized_trend = trend / first_price

        avg_return = state.get("avg_return", 0)
        volatility = state.get("volatility", 0)

        unrealized_pnl = state.get("unrealized_pnl", 0)
        normalized_unrealized_pnl = unrealized_pnl / first_price

        holding_steps = state.get("holding_steps", 0)
        normalized_holding_steps = holding_steps / 20

        cooldown = state.get("cooldown", 0)
        normalized_cooldown = cooldown / 2

        normalized_balance = balance / 10000

        state_list = normalized_prices + [
            normalized_price_change,
            normalized_trend,
            normalized_momentum,
            avg_return,
            volatility,
            normalized_unrealized_pnl,
            normalized_holding_steps,
            normalized_cooldown,
            normalized_balance,
            position,
        ]

        return torch.tensor([state_list], dtype=torch.float32)

    def choose_action(self, state):
        """
        Epsilon-greedy with valid action control.

        Important:
        Exploration is now biased toward trading activity:
        - If holding a position, prefer sell more than hold.
        - If flat, prefer buy more than hold.
        """

        position = state["position"]
        cooldown = state.get("cooldown", 0)

        if random.random() < self.epsilon:
            if position == 1:
                # Already holding: do not buy again.
                # Bias toward sell so the agent learns exits.
                return random.choice([2, 2, 0])

            if cooldown > 0:
                # During cooldown, stay out of market.
                return 0

            # No position: bias toward buy so the agent learns entries.
            return random.choice([1, 1, 0])

        state_tensor = self.preprocess_state(state)

        with torch.no_grad():
            q_values = self.network(state_tensor)[0]

        # Mask invalid actions during exploitation too
        masked_q_values = q_values.clone()

        if position == 1:
            masked_q_values[1] = -1e9  # cannot buy while holding
        else:
            masked_q_values[2] = -1e9  # cannot sell without position

            if cooldown > 0:
                masked_q_values[1] = -1e9  # cannot buy during cooldown

        return torch.argmax(masked_q_values).item()

    def learn(self, batch):
        """
        Learn from batch with valid next-action masking
        """

        states = []
        targets = []

        for state, action, reward, next_state, done in batch:
            state_tensor = self.preprocess_state(state)
            next_state_tensor = self.preprocess_state(next_state)

            q_values = self.network(state_tensor)

            with torch.no_grad():
                next_q_values = self.target_network(next_state_tensor)[0]

                next_position = next_state.get("position", 0)
                next_cooldown = next_state.get("cooldown", 0)

                masked_next_q_values = next_q_values.clone()

                if next_position == 1:
                    masked_next_q_values[1] = -1e9
                else:
                    masked_next_q_values[2] = -1e9

                    if next_cooldown > 0:
                        masked_next_q_values[1] = -1e9

                max_next_q = torch.max(masked_next_q_values)

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