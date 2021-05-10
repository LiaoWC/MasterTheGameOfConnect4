# Reference: https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color
import matplotlib.pyplot as plt
import argparse
import itertools
import numpy as np

# Settings
MARKER = 's'
MARKER_SIZE = 1000
COLOR_MAP = 'YlOrRd'
DEBUG_MODE = False

if __name__ == '__main__':
    if not DEBUG_MODE:
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--file', help='File to plot. '
                                                 'There must be 1+36 value in each of the 3 lines in this file. '
                                                 'First value in each line is the line\' name without any spaces. '
                                                 'Each value split by a space.', required=True)
        args = parser.parse_args()

        with open(args.file, 'r') as file:
            content = str(file.read())

        if content == "":
            raise ValueError("The file contains no content or it's failed to read file.")

        lines = [line for line in content.split('\n') if line != ""]
        names = []
        planes = []
        for line in lines:
            line = line.strip()
            plane = [item for item in line.split(' ') if item != ""]
            name = plane[0]
            plane = list(map(float, plane[1:]))
            names.append(name)
            planes.append(np.array(plane).reshape((6, 6)))
    else:
        # For debugging and testing:
        planes = [np.arange(36.).reshape((6, 6)) for _ in range(3)]
        names = [f'PLOT_2D_DISTRIBUTION_TEST_{i}' for i in range(3)]

    print(planes)
    print(names)
    for plane, name in zip(planes, names):
        c_data = plane.flatten().tolist()
        x_y_pairs = list(itertools.product(np.arange(6), np.arange(6)))
        x_data = [xy[0] for xy in x_y_pairs]
        y_data = [xy[1] for xy in x_y_pairs]

        c_data_normalized = []

        plt.figure(figsize=(8, 8))

        plt.scatter(x=x_data,
                    y=y_data,
                    c=c_data,
                    marker=MARKER,
                    cmap=COLOR_MAP,
                    s=MARKER_SIZE)

        for i, c in enumerate(c_data):
            plt.annotate('{:.2f}'.format(c), (x_data[i], y_data[i]), ha='center')

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.title(name)
        plt.show()
        plt.savefig('{}.png'.format(name))
