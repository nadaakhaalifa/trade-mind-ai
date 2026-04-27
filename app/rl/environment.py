class TradingEnvironment:
    def __init__(self, prices, initial_balance=10000):
        self.prices = prices
        self.initial_balance = initial_balance

        # Trading settings
        self.window_size = 5
        self.position_size = 10
        self.trade_cost = 0.05

        # Risk management
        self.stop_loss = -0.50
        self.take_profit = 0.70
        self.max_holding_steps = 20
        self.cooldown_steps = 2

        self.reset()

    def reset(self):
        self.step_index = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.entry_step = None
        self.cooldown = 0

        return self._get_state()

    def _get_state(self):
        safe_step_index = min(self.step_index, len(self.prices) - 1)

        window = self.prices[
            safe_step_index - self.window_size:safe_step_index
        ]

        current_price = self.prices[safe_step_index]

        first_price = window[0]
        last_price = window[-1]

        price_change = window[-1] - window[-2]
        momentum = last_price - first_price
        moving_avg = sum(window) / len(window)
        trend = last_price - moving_avg

        returns = []
        for i in range(1, len(window)):
            previous_price = window[i - 1]
            current_window_price = window[i]
            returns.append(
                (current_window_price - previous_price) / (previous_price + 1e-9)
            )

        avg_return = sum(returns) / len(returns)
        volatility = sum(abs(r - avg_return) for r in returns) / len(returns)

        unrealized_pnl = 0
        holding_steps = 0

        if self.position == 1:
            unrealized_pnl = current_price - self.entry_price
            holding_steps = safe_step_index - self.entry_step

        return {
            "prices": window,
            "balance": self.balance,
            "position": self.position,

            # Production-level state features
            "price_change": price_change,
            "momentum": momentum,
            "moving_avg": moving_avg,
            "trend": trend,
            "avg_return": avg_return,
            "volatility": volatility,
            "unrealized_pnl": unrealized_pnl,
            "holding_steps": holding_steps,
            "cooldown": self.cooldown,
        }

    def _build_info(self, action, executed_action, current_price, reward):
        return {
            "chosen_action": action,
            "effective_action": action,
            "executed_action": executed_action,
            "price": current_price,
            "reward": reward,
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_step": self.entry_step,
            "cooldown": self.cooldown,
        }

    def step(self, action):
        current_price = self.prices[self.step_index]
        previous_price = self.prices[self.step_index - 1]

        price_change = current_price - previous_price

        reward = 0
        done = False
        executed_action = "hold"

        if self.cooldown > 0:
            self.cooldown -= 1

        # 🛑 STOP LOSS
        if self.position == 1:
            unrealized = current_price - self.entry_price

            if unrealized <= self.stop_loss:
                net = unrealized * self.position_size

                reward = net - self.trade_cost

                # Cap stop-loss punishment so the agent does not become afraid to trade
                reward = max(reward, -2.0)

                self.balance += reward

                self.position = 0
                self.entry_price = 0
                self.entry_step = None
                self.cooldown = self.cooldown_steps

                executed_action = "stop_loss"

                self.step_index += 1

                info = self._build_info(action, executed_action, current_price, reward)
                return self._get_state(), reward, done, info

            # 🎯 TAKE PROFIT
            if unrealized >= self.take_profit:
                net = unrealized * self.position_size

                reward = net * 1.2 - self.trade_cost

                if net > 0:
                    reward += 0.5

                self.balance += reward

                self.position = 0
                self.entry_price = 0
                self.entry_step = None
                self.cooldown = self.cooldown_steps

                executed_action = "take_profit"

                self.step_index += 1

                info = self._build_info(action, executed_action, current_price, reward)
                return self._get_state(), reward, done, info

            # ⏱ MAX HOLDING TIME
            holding_steps = self.step_index - self.entry_step

            if holding_steps >= self.max_holding_steps:
                profit = current_price - self.entry_price
                net = profit * self.position_size

                reward = net * 1.2 - self.trade_cost

                if net > 0:
                    reward += 0.5

                reward = max(reward, -2.0)

                self.balance += reward

                self.position = 0
                self.entry_price = 0
                self.entry_step = None
                self.cooldown = self.cooldown_steps

                executed_action = "max_holding_exit"

                self.step_index += 1

                info = self._build_info(action, executed_action, current_price, reward)
                return self._get_state(), reward, done, info

        # 🔴 HOLD
        if action == 0:
            if self.position == 1:
                # Reward holding winners, but keep it small so reward does not explode
                unrealized = (current_price - self.entry_price) * self.position_size
                reward += unrealized * 0.1
            else:
                # Small penalty for missing market movement while flat
                reward -= abs(price_change) * self.position_size * 0.05

            # Small time cost
            reward -= 0.02
            executed_action = "hold"

        # 🟢 BUY
        elif action == 1:
            if self.cooldown > 0:
                reward -= 0.5
                executed_action = "cooldown_buy_blocked"

            elif self.position == 0:
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

                # Reward equals real PnL with mild profit signal
                reward += net * 1.2 - self.trade_cost

                if net > 0:
                    reward += 0.5

                reward = max(reward, -2.0)

                self.balance += reward

                self.position = 0
                self.entry_price = 0
                self.entry_step = None
                self.cooldown = self.cooldown_steps

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

                reward += net * 1.2 - self.trade_cost

                if net > 0:
                    reward += 0.5

                reward = max(reward, -2.0)

                self.balance += reward

                self.position = 0
                self.entry_price = 0
                self.entry_step = None

                executed_action = "forced_close"

        self.step_index += 1

        info = self._build_info(action, executed_action, current_price, reward)
        return self._get_state(), reward, done, info