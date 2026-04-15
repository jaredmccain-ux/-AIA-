import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_path(path):
    path = np.array(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Pipe Path')
    plt.legend()
    plt.show()