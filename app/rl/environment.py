class TradingEnvironment:
    def __init__(self, prices, initial_balance=10000):
        self.prices = prices
        self.initial_balance = initial_balance

        # Trading settings
        self.window_size = 5
        self.position_size = 10
        self.trade_cost = 0.05

        self.reset()

    def reset(self):
        self.step_index = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.entry_step = None

        return self._get_state()

    def _get_state(self):
        window = self.prices[self.step_index - self.window_size:self.step_index]

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
            # Very small penalty only, so HOLD is allowed when market is bad
            reward -= 0.01
            executed_action = "hold"

        # 🟢 BUY
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.entry_step = self.step_index

                # No fake buy reward. Buying itself is not profit.
                reward -= self.trade_cost
                executed_action = "buy"
            else:
                reward -= 1.0
                executed_action = "invalid_buy"

        # 🔵 SELL
        elif action == 2:
            if self.position == 1:
                profit = current_price - self.entry_price
                net = profit * self.position_size

                # Reward equals real PnL minus small trading cost
                reward += net - self.trade_cost
                self.balance += net - self.trade_cost

                self.position = 0
                self.entry_price = 0
                self.entry_step = None

                executed_action = "sell"
            else:
                reward -= 1.0
                executed_action = "invalid_sell"

        # ⛔ End episode
        if self.step_index >= len(self.prices) - 2:
            done = True

            if self.position == 1:
                profit = current_price - self.entry_price
                net = profit * self.position_size

                reward += net - self.trade_cost
                self.balance += net - self.trade_cost

                self.position = 0
                self.entry_price = 0
                self.entry_step = None

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
            "entry_step": self.entry_step,
            "cooldown": 0,
        }

        return self._get_state(), reward, done, info