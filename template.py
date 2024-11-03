import knu_rl_env
import knu_rl_env.grid_adventure
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import sleep

class GridAdventureRLAgent(knu_rl_env.grid_adventure.GridAdventureAgent):
    def __init__(self, y, x):
        self.q = np.zeros((26, 26, 6))

        self.y = y
        self.x = x

        self.v = np.zeros((26, 26))

        self.r = 0 # 0: 오른쪽방향, 1: 아래방향, 2: 왼쪽방향, 3: 위방향
        self.r_name = ["AR", "AD", "AL", "AU"]
        self.blue_key = 0
        self.red_key = 0
        self.green_key = 0

        self.blue_door = 0
        self.red_door = 0
        self.green_door = 0

        # 키와 문의 사용 여부를 추적하기 위한 변수 추가
        self.used_blue_key = False
        self.used_red_key = False
        self.used_green_key = False
        self.opened_blue_door = False
        self.opened_red_door = False
        self.opened_green_door = False

    def reset(self, y, x):
        self.y = y
        self.x = x

        self.r = 0

        self.v = np.zeros((26, 26))

        self.blue_key = 0
        self.red_key = 0
        self.green_key = 0

        self.blue_door = 0
        self.red_door = 0
        self.green_door = 0

        # 리셋 시 사용 여부도 초기화
        self.used_blue_key = False
        self.used_red_key = False
        self.used_green_key = False
        self.opened_blue_door = False
        self.opened_red_door = False
        self.opened_green_door = False

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
            self.y, self.x = new_loc[0][0], new_loc[1][0]
            self.v[self.y, self.x] += 1
            return


        if cmd == 3: # 키 줍기
            if self.blue_key == 1 or self.red_key == 1 or self.green_key == 1:
                return

            k = np.where(new_state == "KB")
            if not k:
                self.blue_key = 1
                return

            k = np.where(new_state == "KR")
            if not k:
                self.red_key = 1
                return

            k = np.where(new_state == "KG")
            if not k:
                self.green_key = 1
                return

        if cmd == 4: # 키 버리기
            if self.blue_key == 0 and self.red_key == 0 and self.green_key == 0:
                # 아무런 키 없으면 그냥 return
                return

            if self.blue_key == 1:
                k = np.where(new_state == "KB")
                if k:
                    self.blue_key = 0
                return
            if self.red_key == 1:
                k = np.where(new_state == "KR")
                if k:
                    self.red_key = 0
                return
            if self.green_key == 1:
                k = np.where(new_state == "KG")
                if k:
                    self.green_key = 0
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

    def get_reward(self, action, new_state):
        reward = -0.1  # 기본 보상
        if self.v[self.y, self.x] > 1:
            reward = -0.3
        if self.v[self.y, self.x] > 3:
            reward = -0.5
        # 키 줍기 동작(action 3)에 대한 보상
        if action == 3:
            # 파란 키
            if not self.used_blue_key and self.blue_key == 1:
                self.used_blue_key = True
                return 7000
            # 빨간 키
            elif not self.used_red_key and self.red_key == 1:
                self.used_red_key = True
                return 5000
            # 초록 키
            elif not self.used_green_key and self.green_key == 1:
                self.used_green_key = True
                return 5000

        # 문 열기 동작(action 5)에 대한 보상
        if action == 5:
            # 파란 문
            if not self.opened_blue_door and self.blue_door == 1:
                self.opened_blue_door = True
                return 10000
            # 빨간 문
            elif not self.opened_red_door and self.red_door == 1:
                self.opened_red_door = True
                return 10000
            # 초록 문
            elif not self.opened_green_door and self.green_door == 1:
                self.opened_green_door = True
                return 10000
        return reward

    def act(self, state):
        return np.argmax(self.q[self.y, self.x])


def train(episodes, show):
    env = knu_rl_env.grid_adventure.make_grid_adventure(
        show_screen=show,
    )
    agent = GridAdventureRLAgent(1, 1)

    learning_rate_a = 0.4
    discount_factor_g = 0.8
    epsilon = 1.0
    rng = np.random.default_rng()

    # Epsilon 감소 관련 파라미터
    epsilon_start = 1.0
    epsilon_middle = 0.3
    epsilon_end = 0.01
    exploration_episodes = 70
    exploitation_episodes = 30

    reward_episodes = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()[0]
        agent.reset(1, 1)
        record_reward = 0
        terminated = False
        truncated = False
        count = 0
        while not terminated:
            if rng.random() < epsilon:
                action = rng.integers(0, 6) # 무작위 행동
            else:
                action = agent.act(state) # 최적 행동

            y, x = agent.y, agent.x # 이전 위치 저장
            new_state, base_reward, terminated, truncated, _ = env.step(action)
            agent.command(action, state, new_state) # agent 상태 업데이트
            ny, nx = agent.y, agent.x # 현재 위치 저장

            # 보상 계산
            reward = agent.get_reward(action, new_state)

            # 용암에 빠진 여부
            if terminated:
                reward = -33333

            # 목적지 도착 여부
            if ny == 24 and nx == 24:
                reward = 333333

            # Q-table 업데이트
            agent.q[y, x, action] = agent.q[y, x, action] + learning_rate_a * (
                reward + discount_factor_g * np.max(agent.q[ny, nx]) - agent.q[y, x, action]
            )

            state = new_state
            record_reward += reward
            count += 1

            if count > 20000:
                break
        if i < exploration_episodes:
            # 초기 5000 에피소드: 1.0에서 0.3까지 천천히 감소
            epsilon = epsilon_start - (epsilon_start - epsilon_middle) * (i / exploration_episodes)
        else:
            # 5000 에피소드 이후: 0.3에서 0.01까지 빠르게 감소
            remaining_episodes = exploitation_episodes
            current_episode = i - exploration_episodes
            epsilon = max(epsilon_middle * np.exp(-5 * current_episode / remaining_episodes), epsilon_end)

        if i > exploration_episodes:
            learning_rate_a = max(0.3 * np.exp(-3 * (i - exploration_episodes) / exploitation_episodes), 0.01)

        reward_episodes[i] = record_reward
        print(f"Episode {i}: Reward = {record_reward:.2f}, Epsilon = {epsilon:.4f}, Y={agent.y}, X={agent.x}")
        if i % 10 == 0:
            f = open("grid_adventure.pkl", "wb")
            pickle.dump(agent.q, f)
            f.close()

    env.close()

    f = open("grid_adventure.pkl", "wb")
    pickle.dump(agent.q, f)
    f.close()

    plt.plot(reward_episodes)
    plt.savefig("grid_adventure.png")

if __name__ == '__main__':
    # train(100, False)
    agent = GridAdventureRLAgent(1, 1)
    f = open("grid_adventure.pkl", "rb")
    agent.q = pickle.load(f)
    f.close()
    knu_rl_env.grid_adventure.evaluate(agent)

