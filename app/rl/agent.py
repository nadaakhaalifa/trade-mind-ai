import random

# like gambling agent chooses randomly
# Because in RL, agents usually start by exploring.
class DQNAgent:
    def __init__(self, action_size=3):
        """
        action_size: number of possible actions.
        In our case:
        0 = hold
        1 = buy
        2 = sell
        """
        self.action_size = action_size

    def choose_action(self, state):
        """
        Choose an action for the current state.

        For now, the agent acts randomly.
        Later, i will replace this with a neural network.
        """
        return random.randint(0, self.action_size - 1)