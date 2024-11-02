import knu_rl_env
import knu_rl_env.grid_adventure
import numpy as np

class GridAdventureRLAgent(knu_rl_env.grid_adventure.GridAdventureAgent):
    def act(self, state, q):
        player = findPlayer(state)
        action_values = q[player]
        return np.argmax(action_values)


def findPlayer(state):
    mask = np.char.startswith(state, 'A')
    pos = np.argwhere(mask)[0]
    return pos[0], pos[1]

def train(episodes, is_train, show):
    # initialize environment and agent
    env = knu_rl_env.grid_adventure.make_grid_adventure(
        show_screen=show
    )
    agent = GridAdventureRLAgent()

    # load Q-table or initialize
    q = np.zeros((26, 26, 6))

    # hyperparameters
    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1.0     # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    min_epsilon = 0.01
    rng = np.random.default_rng()   # random number generator

    # train
    for i in range(episodes):
        pre_state = env.reset()[0]
        y, x = 1, 1
        total_reward = 0

        terminated = False
        truncated = False
        while not terminated and not truncated:
        # while not terminated:
            if rng.random() > epsilon:  # epsilon이 작을수록 exploitation
                action = np.argmax(q[y, x])
            else:  # epsilon이 클수록 exploration
                action = rng.integers(0, 6)

            new_state, reward, terminated, truncated, _ = env.step(action)
            if action == 2:
                ny, nx = findPlayer(new_state)
                # print(f"Player moved to {ny}, {nx}")
            else:
                ny, nx = y, x

            q[y, x, action] = q[y, x, action] + learning_rate_a * (
                reward + discount_factor_g * np.max(q[ny, nx]) - q[y, x, action]
            )

            y, x = ny, nx
            total_reward += reward

        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
        print(f"Episode {i}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    env.close()


if __name__ == '__main__':
    agent = '''Specify your learned agent'''
    try:
        knu_rl_env.grid_adventure.evaluate(agent)
    except:
        pass
    train(100, True, False)
