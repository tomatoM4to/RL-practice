import knu_rl_env
import knu_rl_env.grid_adventure
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

class ImprovedGridAdventureRLAgent(knu_rl_env.grid_adventure.GridAdventureAgent):
    def __init__(self, y, x):
        # 상태 공간 확장: 위치, 방향, 키 보유 상태를 포함
        self.q = np.random.uniform(0, 0.01, (26, 26, 4, 8, 6))  # y, x, direction, key_state, action

        self.y = y
        self.x = x

        self.r = 0
        self.r_name = ["AR", "AD", "AL", "AU"]
        self.blue_key = 0
        self.red_key = 0
        self.green_key = 0

        self.before_distance = 0

        self.blue_door = 0
        self.red_door = 0
        self.green_door = 0

        # 경로 기억을 위한 변수 추가
        self.visited = np.zeros((26, 26))
        self.path_memory = []

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
        self.blue_key = 0
        self.red_key = 0
        self.green_key = 0

        self.blue_door = 0
        self.red_door = 0
        self.green_door = 0
        self.before_distance = 0
        self.visited = np.zeros((26, 26))
        self.path_memory = []

        self.used_blue_key = False
        self.used_red_key = False
        self.used_green_key = False
        self.opened_blue_door = False
        self.opened_red_door = False
        self.opened_green_door = False

    def rotate(self, cmd):
        if cmd == 0:  # 왼쪽
            self.r -= 1
            if self.r == -1:
                self.r = 3
        if cmd == 1:  # 오른쪽
            self.r += 1
            if self.r == 4:
                self.r = 0

    def command(self, cmd, state, new_state):
        if cmd == 0 or cmd == 1:
            self.rotate(cmd)
            return

        if cmd == 2:  # 앞으로 이동
            r = self.r_name[self.r]
            new_loc = np.where(new_state == r)
            if len(new_loc[0]) > 0 and len(new_loc[1]) > 0:
                self.y, self.x = new_loc[0][0], new_loc[1][0]
            return

        if cmd == 3:  # 키 줍기
            if self.blue_key == 1 or self.red_key == 1 or self.green_key == 1:
                return

            k = np.where(new_state == "KB")
            if len(k[0]) == 0:
                self.blue_key = 1
                return

            k = np.where(new_state == "KR")
            if len(k[0]) == 0:
                self.red_key = 1
                return

            k = np.where(new_state == "KG")
            if len(k[0]) == 0:
                self.green_key = 1
                return

        if cmd == 4:  # 키 버리기
            if self.blue_key == 0 and self.red_key == 0 and self.green_key == 0:
                return

            if self.blue_key == 1:
                k = np.where(new_state == "KB")
                if len(k[0]) > 0:
                    self.blue_key = 0
                return
            if self.red_key == 1:
                k = np.where(new_state == "KR")
                if len(k[0]) > 0:
                    self.red_key = 0
                return
            if self.green_key == 1:
                k = np.where(new_state == "KG")
                if len(k[0]) > 0:
                    self.green_key = 0
                return

        if cmd == 5:  # 문 열기
            if self.blue_key == 1:
                d = np.where(new_state == "DBL")
                if len(d[0]) == 0:
                    self.blue_door += 1
                    return
            if self.red_key == 1:
                d = np.where(new_state == "DRL")
                if len(d[0]) == 0:
                    self.red_door += 1
                    return
            if self.green_key == 1:
                d = np.where(new_state == "DGL")
                if len(d[0]) == 0:
                    self.green_door += 1
                    return

    def get_key_state(self):
        # 키 보유 상태를 8가지 조합으로 인코딩
        return (self.blue_key << 2) | (self.red_key << 1) | self.green_key

    def get_state_index(self):
        return (self.y, self.x, self.r, self.get_key_state())

    def real_distance(self, y, x, new_state):
        q = deque([(y, x)])
        m = np.zeros((26, 26))
        m[y, x] = 1
        dx = [0, 0, -1, 1]
        dy = [-1, 1, 0, 0]
        while q:
            y, x = q.popleft()
            for i in range(4):
                ny = y + dy[i]
                nx = x + dx[i]
                if 0 <= ny and ny < 26 and 0 <= nx and nx < 26 and new_state[ny, nx] != "W" and m[ny, nx] == 0:
                    m[ny, nx] = m[y, x] + 1
        return m[24, 24]

    def get_reward(self, action, new_state):
        base_reward = -0.01

        # 방문 횟수에 기반한 탐험 보상
        visit_penalty = -0.05 * self.visited[self.y, self.x]
        base_reward += visit_penalty

        # 목표지점까지의 거리에 기반한 보상
        distance = self.real_distance(self.y, self.x, new_state)
        if distance < self.before_distance:
            base_reward += 1.0
        else:
            base_reward -= 0.5
        self.before_distance = distance

        # 키 관련 보상
        if action == 3:  # 키 줍기
            if not self.used_blue_key and self.blue_key == 1:
                self.used_blue_key = True
                return 30
            elif not self.used_red_key and self.red_key == 1:
                self.used_red_key = True
                return 30
            elif not self.used_green_key and self.green_key == 1:
                self.used_green_key = True
                return 30

        # 문 열기 보상
        if action == 5:
            if not self.opened_blue_door and self.blue_door == 1:
                self.opened_blue_door = True
                return 50
            elif not self.opened_red_door and self.red_door == 1:
                self.opened_red_door = True
                return 50
            elif not self.opened_green_door and self.green_door == 1:
                self.opened_green_door = True
                return 50

        return base_reward

    def act(self, state):
        state_index = self.get_state_index()

        # 현재 위치 방문 횟수 증가
        self.visited[self.y, self.x] += 1

        # Q-table에서 최적 행동 선택
        return np.argmax(self.q[state_index])

def train(episodes, show):
    env = knu_rl_env.grid_adventure.make_grid_adventure(
        show_screen=show
    )
    agent = ImprovedGridAdventureRLAgent(1, 1)

    learning_rate_a = 0.2
    discount_factor_g = 0.95  # 감가율 증가
    epsilon = 1.0
    rng = np.random.default_rng()

    # 적응적 학습률과 입실론
    min_learning_rate = 0.01
    epsilon_start = 1.0
    epsilon_end = 0.01

    reward_episodes = np.zeros(episodes)
    best_reward = float('-inf')

    for i in range(episodes):
        state = env.reset()[0]
        agent.reset(1, 1)
        agent.visited = np.zeros((26, 26))  # 방문 기록 초기화

        record_reward = 0
        terminated = False
        truncated = False
        step_count = 0

        while not terminated and step_count < 15000:  # 최대 스텝 수 제한
            step_count += 1

            # 입실론-탐욕적 행동 선택
            if rng.random() < epsilon:
                action = rng.integers(0, 6)
            else:
                action = agent.act(state)

            y, x = agent.y, agent.x
            new_state, base_reward, terminated, truncated, _ = env.step(action)
            agent.command(action, state, new_state)
            ny, nx = agent.y, agent.x

            reward = agent.get_reward(action, new_state)

            if terminated:
                reward = -100
            if ny == 24 and nx == 24:
                reward = 500

            # Q-learning 업데이트
            current_state_index = (y, x, agent.r, agent.get_key_state())
            next_state_index = (ny, nx, agent.r, agent.get_key_state())
            agent.q[current_state_index][action] = agent.q[current_state_index][action] + learning_rate_a * (
                reward + discount_factor_g * np.max(agent.q[next_state_index]) - agent.q[current_state_index][action]
            )

            state = new_state
            record_reward += reward

        # 적응적 파라미터 조정
        if i < 2000:  # 처음 3000 에피소드는 천천히 감소
            epsilon = epsilon_start - (epsilon_start - 0.1) * (i / 2000)
        else:  # 나머지 2000 에피소드에서 0.01까지 감소
            epsilon = 0.1 - (0.1 - epsilon_end) * ((i - 2000) / 1000)
        learning_rate_a = max(min_learning_rate, 0.2 * (1 - i/3000))
        reward_episodes[i] = record_reward

        print(f"Episode {i}: Reward = {record_reward:.2f}, Epsilon = {epsilon:.4f}, Steps = {step_count}, Y={agent.y}, X={agent.x}")

        if i % 100 == 0:
            plt.clf()
            plt.plot(reward_episodes)
            plt.savefig("grid_adventure.png")
            f = open("grid_adventure.pkl", "wb")
            pickle.dump(agent.q, f)
            f.close()

    env.close()
    plt.clf()
    plt.plot(reward_episodes)
    plt.savefig("grid_adventure.png")
    f = open("grid_adventure.pkl", "wb")
    pickle.dump(agent.q, f)
    f.close()

if __name__ == '__main__':
    train(3000, False)
    # agent = ImprovedGridAdventureRLAgent(1, 1)
    # f = open("grid_adventure.pkl", "rb")
    # agent.q = pickle.load(f)
    # f.close()
    # knu_rl_env.grid_adventure.evaluate(agent)
