from app.rl.agent import DQNAgent
from app.rl.environment import TradingEnvironment
from app.rl.replay_buffer import ReplayBuffer

# episode 1 → agent practiced once
# total_reward → total profit from that episode
# final_balance → starting balance + profit
# memory_size → saved experiences

class DQNTrainer:
    def __init__(self, prices, episodes=10):
        """
        prices: list of market prices used for training
        episodes: how many times the agent practices
        """
        self.prices = prices
        self.episodes = episodes

        self.env = TradingEnvironment(prices=prices)
        self.agent = DQNAgent()
        self.replay_buffer = ReplayBuffer()

    def train(self):
        """
        Run the training loop.
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

                state = next_state
                total_reward += reward

            results.append({
                "episode": episode + 1,
                "total_reward": total_reward,
                "final_balance": self.env.balance,
                "memory_size": len(self.replay_buffer),
            })

        return results