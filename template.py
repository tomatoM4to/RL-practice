import knu_rl_env
import knu_rl_env.grid_adventure
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class GridAdventureRLAgent(knu_rl_env.grid_adventure.GridAdventureAgent):
    def __init__(self, y, x):
        self.q = np.zeros((26, 26, 6))

        self.y = y
        self.x = x
        self.r = 0 # 0: 오른쪽방향, 1: 아래방향, 2: 왼쪽방향, 3: 위방향
        self.blue_key = 0
        self.red_key = 0
        self.green_key = 0

        self.blue_door = 0
        self.red_door = 0
        self.green_door = 0

    def drop_key(self, key, state):
        if self.r == 0:
            s = state[self.y][self.x + 1]
            if s != "E":
                return
        elif self.r == 1:
            s = state[self.y + 1][self.x]
            if s != "E":
                return
        elif self.r == 2:
            s = state[self.y][self.x - 1]
            if s != "E":
                return
        elif self.r == 3:
            s = state[self.y - 1][self.x]
            if s != "E":
                return

        if key == "B":
            self.blue_key -= 1
        elif key == "R":
            self.red_key -= 1
        else:
            self.green_key -= 1
        return

    def rotate(self, cmd):
        if cmd == 0: # 오른쪽으로 회전
            self.r += 1
            if self.r == 4:
                self.r = 0
        if cmd == 1: # 왼쪽으로 회전
            self.r -= 1
            if self.r == -1:
                self.r = 3

    def command(self, cmd, state):
        if cmd == 0 or cmd == 1:
            self.rotate(cmd)
            return

        if cmd == 2: # 앞으로 이동
            tmp_y, tmp_x = [self.y, self.x]
            if self.r == 0:
                self.x += 1
            elif self.r == 1:
                self.y += 1
            elif self.r == 2:
                self.x -= 1
            else:
                self.y -= 1

            l = state[self.y][self.x]
            if l == "W" or l == "DBL" or l == "DBC" or l == "DGL" or l == "DGC" or l == "DRL" or l == "DRC" or l == "KB" or l == "KR" or l == "KG":
                self.y, self.x = tmp_y, tmp_x
            return


        if cmd == 3: # 열쇠 줍기
            # 파란키
            if self.r == 0 and state[self.y][self.x + 1] == "KB":
                self.blue_key += 1
            if self.r == 1 and state[self.y + 1][self.x] == "KB":
                self.blue_key += 1
            if self.r == 2 and state[self.y][self.x - 1] == "KB":
                self.blue_key += 1
            if self.r == 3 and state[self.y - 1][self.x] == "KB":
                self.blue_key += 1

            # 빨간키
            if self.r == 0 and state[self.y][self.x + 1] == "KR":
                self.red_key += 1
            if self.r == 1 and state[self.y + 1][self.x] == "KR":
                self.red_key += 1
            if self.r == 2 and state[self.y][self.x - 1] == "KR":
                self.red_key += 1
            if self.r == 3 and state[self.y - 1][self.x] == "KR":
                self.red_key += 1

            # 초록키
            if self.r == 0 and state[self.y][self.x + 1] == "KG":
                self.green_key += 1
            if self.r == 1 and state[self.y + 1][self.x] == "KG":
                self.green_key += 1
            if self.r == 2 and state[self.y][self.x - 1] == "KG":
                self.green_key += 1
            if self.r == 3 and state[self.y - 1][self.x] == "KG":
                self.green_key += 1

            return

        if cmd == 4: # 열쇠 버리기
            if self.blue_key == 1:
                self.drop_key("B")
            elif self.red_key == 1:
                self.drop_key("R")
            elif self.green_key == 1:
                self.drop_key("G")
            return

        if cmd == 5: # 문 열기
            # 열린문
            if self.r == 0 and state[self.y][self.x + 1] == "DBC":
                self.blue_door += 1
                return
            if self.r == 0 and state[self.y][self.x + 1] == "DRC":
                self.red_door += 1
                return
            if self.r == 0 and state[self.y][self.x + 1] == "DGC":
                self.green_door += 1
                return

            if self.r == 1 and state[self.y + 1][self.x] == "DBC":
                self.blue_door += 1
                return
            if self.r == 1 and state[self.y + 1][self.x] == "DRC":
                self.red_door += 1
                return
            if self.r == 1 and state[self.y + 1][self.x] == "DGC":
                self.green_door += 1
                return

            if self.r == 2 and state[self.y][self.x - 1] == "DBC":
                self.blue_door += 1
                return
            if self.r == 2 and state[self.y][self.x - 1] == "DRC":
                self.red_door += 1
                return
            if self.r == 2 and state[self.y][self.x - 1] == "DGC":
                self.green_door += 1
                return

            if self.r == 3 and state[self.y - 1][self.x] == "DBC":
                self.blue_door += 1
                return
            if self.r == 3 and state[self.y - 1][self.x] == "DRC":
                self.red_door += 1
                return
            if self.r == 3 and state[self.y - 1][self.x] == "DGC":
                self.green_door += 1
                return

            # 잠긴문
            if self.blue_key == 0 or self.red_key == 0 or self.green_key == 0:
                return

            if self.r == 0 and self.blue_key and state[self.y][self.x + 1] == "DBL":
                self.blue_door += 1
                return
            if self.r == 0 and self.red_key and state[self.y][self.x + 1] == "DRL":
                self.red_door += 1
                return
            if self.r == 0 and self.green_key and state[self.y][self.x + 1] == "DGL":
                self.green_door += 1
                return

            if self.r == 1 and self.blue_key and state[self.y + 1][self.x] == "DBL":
                self.blue_door += 1
                return
            if self.r == 1 and self.red_key and state[self.y + 1][self.x] == "DRL":
                self.red_door += 1
                return
            if self.r == 1 and self.green_key and state[self.y + 1][self.x] == "DGL":
                self.green_door += 1
                return

            if self.r == 2 and self.blue_key and state[self.y][self.x - 1] == "DBL":
                self.blue_door += 1
                return
            if self.r == 2 and self.red_key and state[self.y][self.x - 1] == "DRL":
                self.red_door += 1
                return
            if self.r == 2 and self.green_key and state[self.y][self.x - 1] == "DGL":
                self.green_door += 1
                return

            if self.r == 3 and self.blue_key and state[self.y - 1][self.x] == "DBL":
                self.blue_door += 1
                return
            if self.r == 3 and self.red_key and state[self.y - 1][self.x] == "DRL":
                self.red_door += 1
                return
            if self.r == 3 and self.green_key and state[self.y - 1][self.x] == "DGL":
                self.green_door += 1
                return

    def act(self, state):
        return np.argmax(self.q[self.y, self.x])


def calculate_reward(current_pos, previous_pos, base_reward):
    if base_reward > 0:
        return base_reward
    goal_pos = [24, 24]
    previous_distance = abs(previous_pos[0] - goal_pos[0]) + abs(previous_pos[1] - goal_pos[1])
    current_distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])

    # 목표를 향해 전진하면 작은 양의 보상, 멀어지면 작은 음의 보상
    if current_distance < previous_distance:
        return 2.0  # 목표에 가까워짐
    elif current_distance > previous_distance:
        return -1.0  # 목표에서 멀어짐

    return -0.4  # 제자리 걸음


def train(episodes, is_train, show):
    env = knu_rl_env.grid_adventure.make_grid_adventure(
        show_screen=show
    )
    agent = GridAdventureRLAgent(1, 1)

    # 하이퍼파라미터 조정
    learning_rate_a = 0.3
    discount_factor_g = 0.99  # 미래 보상에 더 큰 가중치
    epsilon = 1.0
    rng = np.random.default_rng()

    reward_episodes = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()[0]
        agent.y, agent.x = 1, 1
        record_reward = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # epsilon-greedy 정책
            if rng.random() < epsilon:
                action = rng.integers(0, 6)
            else:
                action = agent.act(state)

            # action
            new_state, base_reward, terminated, truncated, _ = env.step(action)
            agent.command(action, state)

            # agent 위치 업데이트
            ny, nx = agent.y, agent.x


            # key 보상
            if agent.blue_key == 1 or agent.red_key == 1 or agent.green_key == 1:
                base_reward = 300

            # 문 보상
            if agent.blue_door == 1 or agent.red_door == 1 or agent.green_door == 1:
                base_reward = 300

            # 거리별 보상
            # shaped_reward = calculate_reward((ny, nx), previous_pos, base_reward)

            # 도착지에 도착하면 큰 보상
            if ny == 24 and nx == 24:
                base_reward = 99999

            # Q-table 업데이트
            agent.q[agent.y, agent.x, action] = agent.q[agent.y, agent.x, action] + learning_rate_a * (
                base_reward + discount_factor_g * np.max(agent.q[ny, nx]) - agent.q[agent.y, agent.x, action]
            )

            state = new_state

        epsilon = max(epsilon // 10, 0.01)

        if epsilon <= 0:
            learning_rate_a = 0.0001

        reward_episodes[i] = record_reward
        print(f"Episode {i + 1}: {record_reward}")

    env.close()

    f = open("grid_adventure.pkl", "wb")
    pickle.dump(agent.q, f)
    f.close()

    plt.plot(reward_episodes)
    plt.savefig("grid_adventure.png")

if __name__ == '__main__':
    train(10, True, False)
    # knu_rl_env.grid_adventure.run_manual()
    # knu_rl_env.grid_adventure.evaluate()

