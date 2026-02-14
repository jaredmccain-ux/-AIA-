# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

try:
    matplotlib.use('TkAgg')
except Exception as e:
    print(f"无法设置 TkAgg 后端: {e}")

class Maze:
    def __init__(self):
        self.maze = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ])
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacles = [(1, 1), (1, 3), (2, 3), (3, 0)]

    def get_state(self):
        return self.current_state

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def is_valid_move(self, state):
        row, col = state
        if 0 <= row < 4 and 0 <= col < 4:
            if (row, col) in self.obstacles:
                return False
            return True
        return False

    def step(self, action):
        row, col = self.current_state
        if action == "up":
            new_row = row - 1
            new_col = col
        elif action == "down":
            new_row = row + 1
            new_col = col
        elif action == "left":
            new_row = row
            new_col = col - 1
        elif action == "right":
            new_row = row
            new_col = col + 1
        else:
            return self.current_state, -1, False

        new_state = (new_row, new_col)
        if self.is_valid_move(new_state):
            self.current_state = new_state
            if new_state == self.goal:
                return self.current_state, 10, True
            else:
                return self.current_state, -1, False
        else:
            return self.current_state, -5, False

    def visualize(self, path):
        fig, ax = plt.subplots()
        for row in range(4):
            for col in range(4):
                if (row, col) == self.start:
                    ax.text(col, 3 - row, 'S', ha='center', va='center', fontsize=12, color='green')
                elif (row, col) == self.goal:
                    ax.text(col, 3 - row, 'G', ha='center', va='center', fontsize=12, color='red')
                elif (row, col) in self.obstacles:
                    ax.add_patch(plt.Rectangle((col, 3 - row), 1, 1, facecolor='black'))
                else:
                    ax.add_patch(plt.Rectangle((col, 3 - row), 1, 1, edgecolor='blue', fill=False))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_yticks(np.arange(0, 4, 1))
        ax.grid(True)
        for i in range(len(path)):
            row, col = path[i]
            ax.text(col, 3 - row, f'{i}', ha='center', va='center', fontsize=12, color='orange')
        for i in range(len(path) - 1):
            row1, col1 = path[i]
            row2, col2 = path[i + 1]
            ax.plot([col1, col2], [3 - row1, 3 - row2], 'darkred')
        plt.show()

class QLearning:
    def __init__(self, maze, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.initialize_q_table()
        if seed is not None:
            random.seed(seed)

    def initialize_q_table(self):
        for row in range(4):
            for col in range(4):
                state = (row, col)
                if maze.is_valid_move(state):
                    self.q_table[state] = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.q_table[state].keys()))
        else:
            actions = list(self.q_table[state].items())
            max_q = max(actions, key=lambda x: x[1])[1]
            max_actions = [action for action, q in actions if q == max_q]
            return random.choice(max_actions)

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q - old_q_value)
        self.q_table[state][action] = new_q_value

    def train(self, episodes=10000):
        for episode in range(episodes):
            state = self.maze.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.maze.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def get_optimal_path(self):
        path = [self.maze.reset()]
        while path[-1] != self.maze.goal:
            state = path[-1]
            actions = list(self.q_table[state].items())
            max_q = max(actions, key=lambda x: x[1])[1]
            max_actions = [action for action, q in actions if q == max_q]
            action = random.choice(max_actions)
            next_state, _, _ = self.maze.step(action)
            path.append(next_state)
        return path

maze = Maze()
seed = random.randint(0, 1000)
q_learning = QLearning(maze, epsilon=0.2, seed=seed)
q_learning.train()
optimal_path = q_learning.get_optimal_path()
print("Optimal Path:")
for state in optimal_path:
    print(state)
maze.visualize(optimal_path)