class TradingEnvironment:
    def __init__(
        self,
        prices,
        window_size=5,
        initial_balance=10000,
        trade_fee=0.1,
        hold_reward=0.0,
        invalid_action_penalty=-1.0,
        unrealized_profit_weight=0.01,
        holding_penalty_weight=0.005,
        sell_reward_multiplier=2.0,
        sell_bonus=0.1,
    ):
        """
        prices: list of market prices
        window_size: number of past prices visible to agent
        initial_balance: starting capital
        trade_fee: cost applied when selling
        hold_reward: reward for waiting when not holding
        invalid_action_penalty: penalty for invalid repeated actions
        unrealized_profit_weight: reward weight while holding a position
        holding_penalty_weight: penalty that increases the longer the agent holds
        sell_reward_multiplier: makes successful sells more important
        sell_bonus: small bonus for closing a valid trade
        """

        self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.trade_fee = trade_fee
        self.hold_reward = hold_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.unrealized_profit_weight = unrealized_profit_weight
        self.holding_penalty_weight = holding_penalty_weight
        self.sell_reward_multiplier = sell_reward_multiplier
        self.sell_bonus = sell_bonus

        self.reset()

    def reset(self):
        """
        Reset environment at the start of each episode.
        """

        self.current_step = self.window_size
        self.balance = self.initial_balance

        self.position = 0
        self.entry_price = 0
        self.entry_step = None

        self.done = False

        return self._get_state()

    def _get_state(self):
        """
        Return the current observation/state.
        """

        window = self.prices[
            self.current_step - self.window_size:self.current_step
        ]

        price_change = window[-1] - window[-2]
        moving_avg = sum(window) / len(window)

        return {
            "prices": window,
            "price_change": price_change,
            "moving_avg": moving_avg,
            "balance": self.balance,
            "position": self.position,
        }

    def step(self, action):
        """
        Execute one action.

        Actions:
        0 = HOLD
        1 = BUY
        2 = SELL

        Returns:
        next_state, reward, done, info
        """

        current_price = self.prices[self.current_step]
        reward = 0
        executed_action = "hold"

        # ======================
        # HOLD
        # ======================
        if action == 0:
            if self.position == 1:
                unrealized_profit = current_price - self.entry_price

                # Reward profitable open positions.
                reward = unrealized_profit * self.unrealized_profit_weight

                # Penalize holding too long.
                holding_duration = self.current_step - self.entry_step
                reward -= holding_duration * self.holding_penalty_weight
            else:
                reward = self.hold_reward

            executed_action = "hold"

        # ======================
        # BUY
        # ======================
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.entry_step = self.current_step

                reward = 0
                executed_action = "buy"
            else:
                reward = self.invalid_action_penalty
                executed_action = "invalid_buy"

        # ======================
        # SELL
        # ======================
        elif action == 2:
            if self.position == 1:
                profit = current_price - self.entry_price
                net_profit = profit - self.trade_fee

                self.balance += net_profit

                reward = (net_profit * self.sell_reward_multiplier) + self.sell_bonus

                self.position = 0
                self.entry_price = 0
                self.entry_step = None

                executed_action = "sell"
            else:
                reward = self.invalid_action_penalty
                executed_action = "invalid_sell"

        # ======================
        # UNKNOWN ACTION
        # ======================
        else:
            reward = self.invalid_action_penalty
            executed_action = "invalid_action"

        self.current_step += 1

        if self.current_step >= len(self.prices) - 1:
            self.done = True

        info = {
            "chosen_action": action,
            "executed_action": executed_action,
            "price": current_price,
            "reward": reward,
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_step": self.entry_step,
        }

        return self._get_state(), reward, self.done, info