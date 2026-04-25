class TradingEnvironment:
    def __init__(self, prices, window_size=5, initial_balance=10000):
        """
        prices: list of market prices
        window_size: how many previous prices the agent can see
        initial_balance: starting money
        """
        self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        Called at the start of each episode.
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance

        self.position = 0       # 0 = no position, 1 = holding asset
        self.entry_price = 0    # price at which we bought

        self.done = False

        return self._get_state()

    def _get_state(self):
        """
        Return the current observation (what the agent sees).
        """
        window = self.prices[
            self.current_step - self.window_size:self.current_step
        ]

        return {
            "prices": window,
            "balance": self.balance,
            "position": self.position,
        }

    def step(self, action):
        """
        Perform one action in the environment.

        action:
        0 = hold
        1 = buy
        2 = sell

        Returns:
        next_state, reward, done
        """
        current_price = self.prices[self.current_step]
        reward = 0

        # BUY
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price

        # SELL
        elif action == 2 and self.position == 1:
            profit = current_price - self.entry_price
            self.balance += profit
            reward = profit

            self.position = 0
            self.entry_price = 0

        # move forward in time
        self.current_step += 1

        # check if episode finished
        if self.current_step >= len(self.prices) - 1:
            self.done = True

        return self._get_state(), reward, self.done