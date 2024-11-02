import knu_rl_env
import knu_rl_env.grid_adventure
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class GridAdventureRLAgent(knu_rl_env.grid_adventure.GridAdventureAgent):
    def act(self, state, q):
        player = findPlayer(state)
        action_values = q[player]
        return np.argmax(action_values)

def findPlayer(state):
    mask = np.char.startswith(state, 'A')
    pos = np.argwhere(mask)[0]
    return pos[0], pos[1]

def findGoal(state):
    mask = np.char.startswith(state, 'G')
    pos = np.argwhere(mask)[0]
    return pos[0], pos[1]

def calculate_reward(current_pos, previous_pos, base_reward):
    if base_reward > 0:
        return base_reward
    goal_pos = [24, 24]
    previous_distance = abs(previous_pos[0] - goal_pos[0]) + abs(previous_pos[1] - goal_pos[1])
    current_distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])

    # 목표를 향해 전진하면 작은 양의 보상, 멀어지면 작은 음의 보상
    if current_distance < previous_distance:
        return 0.2  # 목표에 가까워짐
    elif current_distance > previous_distance:
        return -0.1  # 목표에서 멀어짐

    return -0.05  # 제자리 걸음

def train(episodes, is_train, show):
    env = knu_rl_env.grid_adventure.make_grid_adventure(
        show_screen=show
    )
    agent = GridAdventureRLAgent()

    q = np.zeros((26, 26, 6))

    # 하이퍼파라미터 조정
    learning_rate_a = 0.3
    discount_factor_g = 0.9  # 미래 보상에 더 큰 가중치
    epsilon = 1.0
    epsilon_decay_rate = 0.001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        current_state = env.reset()[0]
        y, x = findPlayer(current_state)
        total_reward = 0

        terminated = False
        truncated = False
        while not terminated and not truncated:
            previous_pos = (y, x)

            # epsilon-greedy 정책
            if rng.random() < epsilon:
                action = rng.integers(0, 6)
            else:
                action = np.argmax(q[y, x])

            new_state, base_reward, terminated, truncated, _ = env.step(action)

            if action == 2:  # 이동 액션인 경우
                ny, nx = findPlayer(new_state)
            else:
                ny, nx = y, x

            # 수정된 보상 계산
            shaped_reward = calculate_reward((ny, nx), previous_pos, base_reward)
            total_reward += shaped_reward

            # Q-learning 업데이트
            q[y, x, action] = q[y, x, action] + learning_rate_a * (
                shaped_reward + discount_factor_g * np.max(q[ny, nx]) - q[y, x, action]
            )

            y, x = ny, nx

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon <= 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = total_reward
        # print(f"Episode {i + 1}: {total_reward}")

    env.close()

    f = open("grid_adventure.pkl", "wb")
    pickle.dump(q, f)
    f.close()

    plt.plot(rewards_per_episode)
    plt.savefig("grid_adventure.png")

if __name__ == '__main__':
    train(5, True, False)
