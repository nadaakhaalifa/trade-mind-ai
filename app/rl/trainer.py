from app.rl.agent import DQNAgent
from app.rl.environment import TradingEnvironment
from app.rl.replay_buffer import ReplayBuffer

# State → Action → Reward → Learn → Improve DQN

class DQNTrainer:
    def __init__(self, prices, episodes=10, batch_size=4):
        """
        prices: market prices used for training
        episodes: how many times the agent practices
        batch_size: how many memories the agent learns from each step
        """
        self.prices = prices
        self.episodes = episodes
        self.batch_size = batch_size

        self.env = TradingEnvironment(prices=prices)
        self.agent = DQNAgent()
        self.replay_buffer = ReplayBuffer()

    def train(self):
        """
        Run training episodes and let the agent learn from stored experiences.
        """
        results = []

        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.env.step(action)

                self.replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.agent.learn(batch)

                state = next_state
                total_reward += reward

            self.agent.decay_epsilon()

            results.append({
                "episode": episode + 1,
                "total_reward": total_reward,
                "final_balance": self.env.balance,
                "memory_size": len(self.replay_buffer),
                "epsilon": self.agent.epsilon,
            })

        return results