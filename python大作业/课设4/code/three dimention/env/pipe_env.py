import numpy as np

class PipeLayoutEnv:
    def __init__(self, space, start, goal):
        self.space = space  # 三维空间
        self.start = start  # 起点
        self.goal = goal    # 终点
        self.current_pos = start
        self.path = [start]  # 已生成的路径

    def step(self, action):
        """执行动作并返回新的状态、奖励、是否完成"""
        x, y, z = self.current_pos
        dx, dy, dz = action
        new_pos = (x + dx, y + dy, z + dz)

        if not (0 <= new_pos[0] < self.space.shape[0] and
                0 <= new_pos[1] < self.space.shape[1] and
                0 <= new_pos[2] < self.space.shape[2]):
            return self.current_pos, -10, False  # 越界惩罚

        if self.space[new_pos] == 1:
            return self.current_pos, -10, False  # 碰撞惩罚

        self.current_pos = new_pos
        self.path.append(new_pos)
        reward = 1 if new_pos != self.goal else 100
        done = new_pos == self.goal
        return new_pos, reward, done

    def reset(self):
        """重置环境"""
        self.current_pos = self.start
        self.path = [self.start]
        return self.start