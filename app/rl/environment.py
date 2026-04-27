class TradingEnvironment:
    def __init__(self, prices, initial_balance=10000):
        self.prices = prices
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.step_index = 5
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0

        return self._get_state()

    def _get_state(self):
        window = self.prices[self.step_index - 5:self.step_index]

        return {
            "prices": window,
            "balance": self.balance,
            "position": self.position
        }

    def step(self, action):
        current_price = self.prices[self.step_index]

        reward = 0
        done = False
        executed_action = "hold"

        # 🔴 HOLD
        if action == 0:
            reward -= 0.2
            executed_action = "hold"

        # 🟢 BUY
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price

                reward += 2.0
                executed_action = "buy"
            else:
                reward -= 1.0
                executed_action = "invalid_buy"

        # 🔵 SELL
        elif action == 2:
            if self.position == 1:
                profit = current_price - self.entry_price
                net = profit * 10

                reward += net * 20

                if profit > 0:
                    reward += 10.0
                else:
                    reward -= 5.0

                self.balance += net

                self.position = 0
                self.entry_price = 0

                executed_action = "sell"
            else:
                reward -= 1.0
                executed_action = "invalid_sell"

        # ⛔ End episode
        if self.step_index >= len(self.prices) - 2:
            done = True

            if self.position == 1:
                profit = current_price - self.entry_price
                net = profit * 10

                reward += net * 20
                self.balance += net

                self.position = 0
                self.entry_price = 0

                executed_action = "forced_close"

        self.step_index += 1

        info = {
               "chosen_action": action,
               "effective_action": action,
               "executed_action": executed_action,
               "price": current_price,
               "reward": reward,
               "balance": self.balance,
               "position": self.position,
               "entry_price": self.entry_price,
               "entry_step": None,
               "cooldown": 0,
        }

        return self._get_state(), reward, done, info