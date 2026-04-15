import heapq
import numpy as np
import matplotlib.pyplot as plt
import random

# 定义网格环境
class GridEnvironment:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles  # 障碍物位置

    def is_valid(self, x, y):
        """判断节点是否有效"""
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def random_free_space(self, num_points):
        """生成指定数量的随机点，确保它们不在障碍物上"""
        free_points = set()
        while len(free_points) < num_points:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.obstacles:
                free_points.add((x, y))
        return list(free_points)

# A* 路径规划
class AStar:
    def __init__(self, environment):
        self.env = environment

    def heuristic(self, x1, y1, x2, y2):
        """曼哈顿距离作为启发式函数"""
        return abs(x1 - x2) + abs(y1 - y2)

    def search(self, start, goal):
        """A* 搜索"""
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(*start, *goal), start))  # (优先级, 节点)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(*start, *goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (x + dx, y + dy)

                if not self.env.is_valid(*neighbor):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(*neighbor, *goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        """重建路径"""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

# 绘制图像
def draw_paths(env, paths, starts, goals, path_labels):
    plt.figure(figsize=(15, 15))
    grid = np.zeros((env.height, env.width))

    # 绘制障碍物
    for x, y in env.obstacles:
        grid[y, x] = 1

    plt.imshow(grid, cmap="Reds", origin="lower", alpha=0.5)  # 使用红色系颜色图表示障碍物

    # 绘制路径
    colors = ['blue', 'green', 'yellow', 'red', 'cyan', 'magenta', 'black', 'grey']
    start_colors = ['orange', 'lightgreen', 'pink', 'gold', 'lightblue', 'violet', 'brown', 'tan']
    goal_colors = ['darkorange', 'forestgreen', 'darkmagenta', 'darkred', 'navy', 'indigo', 'dimgray', 'sienna']
    for i, (path, start, goal) in enumerate(zip(paths, starts, goals)):
        if path:  # 确保路径不为空
            x, y = zip(*path)
            plt.plot(x, y, color=colors[i % len(colors)], linewidth=1, alpha=0.8, label=path_labels[i] if i < len(path_labels) else "")
        plt.scatter(*start, color=start_colors[i % len(start_colors)], s=50, label=f"Start {i+1}" if i == 0 else "")
        plt.scatter(*goal, color=goal_colors[i % len(goal_colors)], s=50, label=f"Goal {i+1}" if i == 0 else "")

    plt.title("Pipe Planning Visualization")
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Paths and Points")
    plt.show()

# 主程序
def main():
    # 定义网格和障碍物
    width, height = 60, 60
    # 障碍物分块在不同区域出现
    obstacles = (
        {(x, y) for x in range(10, 20) for y in range(10, 20)}  # 第一块障碍物区域
        | {(x, y) for x in range(40, 50) for y in range(10, 20)}  # 第二块障碍物区域
        | {(x, y) for x in range(10, 20) for y in range(40, 50)}  # 第三块障碍物区域
        | {(x, y) for x in range(40, 50) for y in range(40, 50)}  # 第四块障碍物区域
        | {(x, y) for x in range(20, 40) for y in range(20, 40)}  # 第五块障碍物区域
    )

    # 初始化环境
    env = GridEnvironment(width, height, obstacles)
    astar = AStar(env)

    # 生成随机起点和终点
    num_paths = 50  # 定义路径数量
    starts = env.random_free_space(num_paths)
    goals = env.random_free_space(num_paths)

    paths = []
    path_labels = [f"Path {i+1}" for i in range(num_paths)]  # 创建路径标签
    for start, goal in zip(starts, goals):
        path = astar.search(start, goal)
        if path:
            paths.append(path)

    # 绘制结果
    draw_paths(env, paths, starts, goals, path_labels)

if __name__ == "__main__":
    main()