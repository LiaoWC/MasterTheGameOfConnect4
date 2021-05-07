import sys
import numpy as np
import random
import mpl_toolkits
import matplotlib.pylab as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# Settings (use matplotlib named colors)
EMPTY_COLOR = 'lightgray'
BLACK_STONE_COLOR = 'darkblue'
WHITE_STONE_COLOR = 'gold'
STONE_SIZE = 100

# Constant
SIDE_LEN = 6
MARKER = 'o'

# Example board
EXAMPLE_BOARD = np.array(
    list(map(
        int,
        '-1 -1 0 1 -1 -1 -1 0 0 0 0 -1 0 0 0 0 0 1 2 0 0 0 0 1 -1 0 0 0 0 -1 -1 -1 0 2 -1 -1 -1 -1 0 1 -1 -1 -1 0 0 0 0'
        ' -1 0 0 0 0 0 2 0 0 0 0 0 1 -1 0 0 0 0 -1 -1 -1 0 0 -1 -1 -1 -1 0 2 -1 -1 -1 0 0 0 0 -1 0 0 0 0 0 1 0 0 0 0 0 '
        '0 -1 0 0 0 0 -1 -1 -1 0 0 -1 -1 -1 -1 0 1 -1 -1 -1 0 0 0 0 -1 0 0 0 0 0 2 0 0 0 0 0 0 -1 0 0 0 0 -1 -1 -1 0 0 '
        '-1 -1 -1 -1 0 2 -1 -1 -1 0 0 0 0 -1 0 0 0 0 0 2 0 0 0 0 0 0 -1 0 0 0 0 -1 -1 -1 0 0 -1 -1 -1 -1 0 2 -1 -1 -1 0'
        ' 0 0 0 -1 0 0 0 0 0 1 0 0 0 0 0 0 -1 0 0 0 0 -1 -1 -1 0 0 -1 -1'.split(' '))))


def generate_random_state() -> "np.ndarray":
    tmp = []
    for i in range(SIDE_LEN * SIDE_LEN * SIDE_LEN):
        tmp.append(random.randint(-1, 2))
    return np.array(tmp).reshape((SIDE_LEN, SIDE_LEN, SIDE_LEN))


if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) <= 1:
        print('Usage: python3 plot_state.py board[0][0][0] board[0][0][1] ... (216 totally)')
        print('You did not provide a state, so we make a random state.')
        state = generate_random_state()
    else:
        # Check argc
        if len(sys.argv) != SIDE_LEN * SIDE_LEN * SIDE_LEN + 1:
            raise ValueError(
                "Number of value provided to form a state must be {}.".format(SIDE_LEN * SIDE_LEN * SIDE_LEN))
        # Gather arguments to make a state
        state = np.array(list(map(int, sys.argv[1:]))).reshape((SIDE_LEN, SIDE_LEN, SIDE_LEN))

    # Use example state?
    state = EXAMPLE_BOARD.reshape((SIDE_LEN, SIDE_LEN, SIDE_LEN))

    # Start to plot
    # Reference: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Remove those empty positions that are on other empty positions (set them to be illegal(-1))
    print(state)
    for x in range(SIDE_LEN):
        for y in range(SIDE_LEN):
            if state[0][x][y] == -1:
                continue
            find_empty = False
            for z in range(SIDE_LEN):
                if find_empty:
                    state[z][x][y] = -1
                elif state[z][x][y] == 0:
                    find_empty = True

    print(state)
    # exit(1)

    # Data for three-dimensional scattered points
    l_data = []
    x_data = []
    y_data = []
    for z, plane in enumerate(state):
        for i, line in enumerate(plane):
            for j, item in enumerate(line):
                l_data.append(z)
                x_data.append(i)
                y_data.append(j)

    l_data = np.array(l_data)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    c_data = list(state.flatten())
    # print(len(l_data))
    # print(l_data)
    # print(x_data)
    # print(y_data)

    # Remove illegal positions
    l_data_remove_illegal = []
    x_data_remove_illegal = []
    y_data_remove_illegal = []
    c_data_remove_illegal = []
    for i, c in enumerate(c_data):
        if c < 0:
            continue
        else:
            l_data_remove_illegal.append(l_data[i])
            x_data_remove_illegal.append(x_data[i])
            y_data_remove_illegal.append(y_data[i])
            c_data_remove_illegal.append(c_data[i])

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')

    ax.scatter3D(x_data_remove_illegal,
                 y_data_remove_illegal,
                 l_data_remove_illegal,
                 c=c_data_remove_illegal,
                 cmap=LinearSegmentedColormap.from_list('connect4_3d',
                                                        [EMPTY_COLOR, BLACK_STONE_COLOR, WHITE_STONE_COLOR]),
                 s=STONE_SIZE,
                 marker=MARKER)
    plt.show()
