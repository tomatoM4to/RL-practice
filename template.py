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
        self.ny = y
        self.nx = x

        self.r = 0 # 0: 오른쪽방향, 1: 아래방향, 2: 왼쪽방향, 3: 위방향
        self.r_name = ["AR", "AD", "AL", "AU"]
        self.blue_key = 0
        self.red_key = 0
        self.green_key = 0

        self.blue_door = 0
        self.red_door = 0
        self.green_door = 0

    def reset(self, y, x):
        self.y = y
        self.x = x
        self.ny = y
        self.nx = x

        self.r = 0
        self.blue_key = 0
        self.red_key = 0
        self.green_key = 0

        self.blue_door = 0
        self.red_door = 0
        self.green_door = 0


    def rotate(self, cmd):
        if cmd == 0: # 왼쪽
            self.r -= 1
            if self.r == -1:
                self.r = 3
        if cmd == 1: # 오른쪽
            self.r += 1
            if self.r == 4:
                self.r = 0

    def command(self, cmd, state, new_state):
        if cmd == 0 or cmd == 1:
            self.rotate(cmd)
            return

        if cmd == 2: # 앞으로 이동
            r = self.r_name[self.r]
            new_loc = np.where(new_state == r)
            self.ny, self.nx = new_loc[0][0], new_loc[1][0]
            # try:
            #     self.ny, self.nx = new_loc[0][0], new_loc[1][0]
            # except:
            #     print(new_loc, r)
            #     print(new_state)
            #     exit(1)


        if cmd == 3: # 키 줍기
            if self.blue_key == 0:
                k = np.where(new_state == "KB")
                if not k:
                    self.blue_key += 1
                return
            if self.red_key == 0:
                k = np.where(new_state == "KR")
                if not k:
                    self.red_key += 1
                return
            if self.green_key == 0:
                k = np.where(new_state == "KG")
                if not k:
                    self.green_key += 1
                return

        if cmd == 4: # 키 버리기
            if self.blue_key == 0 and self.red_key == 0 and self.green_key == 0:
                # 아무런 키 없으면 그냥 return
                return

            if self.blue_key == 1:
                k = np.where(new_state == "KB")
                if k:
                    self.blue_key -= 0
                return
            if self.red_key == 1:
                k = np.where(new_state == "KR")
                if k:
                    self.red_key -= 0
                return
            if self.green_key == 1:
                k = np.where(new_state == "KG")
                if k:
                    self.green_key -= 0
                return

        if cmd == 5: # 문 열기
            if self.blue_key == 1:
                d = np.where(new_state == "DBL")
                if not d:
                    self.blue_door += 1
                    return
            if self.red_key == 1:
                d = np.where(new_state == "DRL")
                if not d:
                    self.red_door += 1
                    return
            if self.green_key == 1:
                d = np.where(new_state == "DGL")
                if not d:
                    self.green_door += 1
                    return

    def distance(self, base_reward):
        if base_reward != 0:
            return base_reward

        old = abs(self.y - 24) + abs(self.x - 24)
        new = abs(self.ny - 24) + abs(self.nx - 24)
        self.y, self.x = self.ny, self.nx
        if new < old:
            return 5.0
        elif new > old:
            return -2.0
        else:
            return -1.0

    def act(self, state):
        return np.argmax(self.q[self.y, self.x])


def train(episodes, is_train, show):
    env = knu_rl_env.grid_adventure.make_grid_adventure(
        show_screen=show
    )
    agent = GridAdventureRLAgent(1, 1)

    # 하이퍼파라미터 조정
    learning_rate_a = 0.3
    discount_factor_g = 0.9  # 미래 보상에 더 큰 가중치
    epsilon = 1.0
    rng = np.random.default_rng()

    reward_episodes = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()[0]
        agent.reset(1, 1)
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
            agent.command(action, state, new_state)
            y, x = agent.y, agent.x
            if terminated:
                print("terminated")
            if truncated:
                print("truncated")
            # key 보상
            if agent.blue_key == 1 or agent.red_key == 1 or agent.green_key == 1:
                base_reward = 50
            if agent.blue_key > 1 and agent.red_key > 1 and agent.green_key > 1:
                base_reward = -50

            # 문 보상
            if agent.blue_door == 1 or agent.red_door == 1 or agent.green_door == 1:
                base_reward = 50
            elif agent.blue_door > 1 and agent.red_door > 1 and agent.green_door > 1:
                base_reward = -30

            base_reward = agent.distance(base_reward)

            # 도착지에 도착하면 큰 보상
            if agent.ny == 24 and agent.nx == 24:
                base_reward = 99999

            # Q-table 업데이트
            agent.q[y, x, action] = agent.q[y, x, action] + learning_rate_a * (
                base_reward + discount_factor_g * np.max(agent.q[agent.ny, agent.nx]) - agent.q[y, x, action]
            )

            state = new_state
            record_reward += base_reward

        epsilon = max(epsilon - 0.001, 0.01)

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
    train(1000, True, False)

