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
        self.q = np.random.uniform(0, 0.01, (26, 26, 6))

        self.y = y
        self.x = x

        self.v = np.zeros((26, 26))
        self.recent_positions = []  # 최근 위치 저장
        self.max_recent = 5

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
        self.recent_positions = []  # 최근 위치 저장
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
        # 시작점과 목표점 정의
        goal_y, goal_x = 24, 24
        current_manhattan = abs(self.y - goal_y) + abs(self.x - goal_x)

        # 기본 보상 설정 (더 큰 페널티)
        reward = -0.1

        # 중복 방문에 대한 강화된 페널티
        if self.v[self.y, self.x] > 1:
            reward -= 0.2  # 페널티 증가
        if self.v[self.y, self.x] > 3:
            reward -= 0.3  # 더 큰 페널티
        if self.v[self.y, self.x] > 5:
            reward -= 0.5  # 매우 큰 페널티로 같은 위치 반복 방문 방지

        # 거리 기반 보상 (보상 크기 증가)
        distance_reward = (48 - current_manhattan) / 48.0
        reward += distance_reward * 0.3  # 거리에 따른 보상 증가

        # 이전 위치로 돌아가는 것에 대한 추가 페널티
        if len(self.recent_positions) > 0 and (self.y, self.x) == self.recent_positions[-1]:
            reward -= 0.2

        # 키 관련 보상 (증가된 보상)
        if action == 3:
            if not self.used_blue_key and self.blue_key == 1:
                self.used_blue_key = True
                return 50 + distance_reward * 10
            elif not self.used_red_key and self.red_key == 1:
                self.used_red_key = True
                return 40 + distance_reward * 10
            elif not self.used_green_key and self.green_key == 1:
                self.used_green_key = True
                return 40 + distance_reward * 10

        # 문 관련 보상 (증가된 보상)
        if action == 5:
            if not self.opened_blue_door and self.blue_door == 1:
                self.opened_blue_door = True
                return 75 + distance_reward * 10
            elif not self.opened_red_door and self.red_door == 1:
                self.opened_red_door = True
                return 75 + distance_reward * 10
            elif not self.opened_green_door and self.green_door == 1:
                self.opened_green_door = True
                return 75 + distance_reward * 10

        # 최근 위치 업데이트
        self.recent_positions.append((self.y, self.x))
        if len(self.recent_positions) > self.max_recent:
            self.recent_positions.pop(0)
        return reward

    def act(self, state):
        return np.argmax(self.q[self.y, self.x])


def train(episodes, show):
    env = knu_rl_env.grid_adventure.make_grid_adventure(
        show_screen=show,
    )
    agent = GridAdventureRLAgent(1, 1)

    # 하이퍼파라미터 조정
    learning_rate = 0.5  # 학습률 증가
    discount_factor = 0.95  # 미래 보상 중요도 증가
    epsilon = 1.0
    min_epsilon = 0.1  # 최소 입실론 값 증가
    epsilon_decay = 0.95
    rng = np.random.default_rng()

    reward_episodes = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()[0]
        agent.reset(1, 1)
        record_reward = 0
        terminated = False
        truncated = False
        steps = 0
        stuck_counter = 0 # 같은 위치에 머무는 시간
        last_position = None

        while not terminated:
            if rng.random() < epsilon:
                if stuck_counter > 5:  # 오래 갇혀있으면 완전 랜덤 행동
                    action = np.random.randint(6)
                else:
                    # Q-value에 노이즈를 추가한 소프트맥스 선택
                    q_values = agent.q[agent.y, agent.x] + np.random.normal(0, 0.1, 6)
                    action = np.argmax(q_values)
            else:
                action = agent.act(state) # 최적 행동

            prev_y, prev_x = agent.y, agent.x # 이전 위치 저장
            new_state, base_reward, terminated, truncated, _ = env.step(action)
            agent.command(action, state, new_state) # agent 상태 업데이트
            ny, nx = agent.y, agent.x # 현재 위치 저장

            # stuck 상태 체크
            current_position = (agent.y, agent.x)
            if current_position == last_position:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_position = current_position

            # 보상 계산
            reward = agent.get_reward(action, new_state)

            if terminated:
                reward = -200  # 실패 페널티 증가
            elif agent.y == 24 and agent.x == 24:
                reward = 500  # 성공 보상 증가
                terminated = True

            # Q-learning 업데이트 with experience replay
            next_max_q = np.max(agent.q[agent.y, agent.x])
            agent.q[prev_y, prev_x, action] = (1 - learning_rate) * agent.q[prev_y, prev_x, action] + \
                                            learning_rate * (reward + discount_factor * next_max_q)

            state = new_state
            record_reward += reward
            steps += 1

            # stuck 상태에서 벗어나기 위한 추가 로직
            if stuck_counter > 10:
                learning_rate = min(0.9, learning_rate * 1.1)  # 학습률 임시 증가
                epsilon = min(1.0, epsilon * 1.1)  # 입실론 임시 증가
            else:
                learning_rate = max(0.1, learning_rate * 0.999)  # 정상 학습률로 복귀
                epsilon = max(min_epsilon, epsilon * epsilon_decay)  # 정상 입실론으로 복귀

            if terminated or steps > 30000:
                break


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

