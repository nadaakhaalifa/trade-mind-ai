import random
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=10000):
        """
        Stores the agent's past experiences.

        max_size: maximum number of experiences to keep in memory
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """
        Add one experience to memory.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly select a group of experiences.
        The agent will learn from this group later.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Return how many experiences are stored.
        """
        return len(self.buffer)