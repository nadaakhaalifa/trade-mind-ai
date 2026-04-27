import random

from app.rl.data_split import split_prices
from app.rl.environment import TradingEnvironment
from app.rl.agent import DQNAgent


def load_prices():
    return [
        100, 101, 102, 101, 103, 105, 104, 106, 108, 107,
        109, 111, 110, 112, 115, 114, 116, 118, 117, 120,
        119, 121, 123, 122, 124, 126, 125, 127, 130, 129,
    ]


def evaluate_agent(agent, prices):
    env = TradingEnvironment(prices)

    state = env.reset()
    done = False
    total_reward = 0

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

    agent.epsilon = old_epsilon

    return {
        "total_reward": total_reward,
        "final_balance": env.balance,
        "trades": getattr(env, "trades_count", None),
    }


def main():
    prices = load_prices()

    train_prices, test_prices = split_prices(prices)

    train_env = TradingEnvironment(train_prices)

    first_state = train_env.reset()
    state_size = len(agent_state_preview(first_state))
    action_size = 3

    agent = DQNAgent(state_size, action_size)

    episodes = 300
    batch_size = 32
    memory = []

    for episode in range(episodes):
        state = train_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = train_env.step(action)

            memory.append((state, action, reward, next_state, done))

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                agent.learn(batch)

            episode_reward += reward
            state = next_state

        agent.decay_epsilon()

        if episode % 10 == 0:
            agent.update_target_network()

        print(
            f"Episode {episode + 1}/{episodes} | "
            f"reward={episode_reward:.2f} | "
            f"epsilon={agent.epsilon:.3f}"
        )

    train_result = evaluate_agent(agent, train_prices)
    test_result = evaluate_agent(agent, test_prices)

    print("\nGENERALIZATION RESULTS")
    print("----------------------")
    print("Train result:", train_result)
    print("Unseen test result:", test_result)

    if test_result["final_balance"] > train_env.initial_balance:
        print("✅ Agent generalized well on unseen data")
    else:
        print("⚠️ Agent did not generalize well yet")


def agent_state_preview(state):
    """
    Builds the same state tensor shape used by DQNAgent.preprocess_state,
    only to calculate state_size safely before creating the agent.
    """

    prices = state["prices"]

    return prices + [
        state.get("price_change", prices[-1] - prices[-2]),
        state.get("trend", 0),
        state.get("momentum", prices[-1] - prices[0]),
        state.get("avg_return", 0),
        state.get("volatility", 0),
        state.get("unrealized_pnl", 0),
        state.get("holding_steps", 0),
        state.get("cooldown", 0),
        state["balance"],
        state["position"],
    ]


if __name__ == "__main__":
    main()