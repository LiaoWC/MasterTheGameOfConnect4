import numpy as np
import random
import math
import time
from numpy.typing import ArrayLike
from typing import List, Sequence, Tuple
from copy import deepcopy
import torch
from abc import ABC
from torch import nn

# the coefficient for ucb
c = 1.414

# global variable for counting all playing times
n = 0


#
class Color:
    BLACK = 1
    WHITE = 2
    EMPTY = 0
    ILLEGAL = -1


class State:
    DEPTH = 6
    HEIGHT = 6
    WIDTH = 6
    STATE_SHAPE = (DEPTH, HEIGHT, WIDTH)


def get_next_possible_move(state):
    possible_moves = []
    for i in range(6):
        for j in range(6):
            if state[5][i][j] != 0:
                continue
            for l in range(6):
                if state[l][i][j] == 0:
                    possible_moves.append([l, i, j])
                    break
    ret = np.array(possible_moves)
    return ret


def ucb(visiting_count, winning_count):
    global c
    global n
    if visiting_count == 0:
        visiting_count = 0.999
    p = math.log(n, 10) / visiting_count
    p = pow(p, 0.5) * c
    p = p + winning_count / visiting_count
    return p


def boundary_test(coordinate):
    for i in range(3):
        if coordinate[i] < 0 or coordinate[i] >= 6:
            return False
    return True


def get_state_properties_b(start_state, start_state_properties, movements):
    dirs = [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, -1],
        [1, 0, -1],
        [1, -1, 0],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1]
    ]

    temp_state = start_state
    temp_properties = start_state_properties
    for move in movements:
        possible_dir = []
        l, i, j, c = move[0], move[1], move[2], move[3]
        for dir in dirs:
            muls = np.arange(1, 4)
            cnt = 1
            for mul in muls:
                temp_coordinate = np.add(move[0:3], np.multiply(dir, mul))
                if not boundary_test(temp_coordinate):
                    break
                temp_l, temp_i, temp_j = \
                    temp_coordinate[0], temp_coordinate[1], temp_coordinate[2]
                if temp_state[temp_l, temp_i, temp_j] != c:
                    break
                cnt += 1
            for mul in muls:
                temp_coordinate = np.add(move[0:3], np.multiply(dir, -mul))
                if not boundary_test(temp_coordinate):
                    break
                temp_l, temp_i, temp_j = \
                    temp_coordinate[0], temp_coordinate[1], temp_coordinate[2]
                if temp_state[temp_l, temp_i, temp_j] != c:
                    break
                cnt += 1
            while cnt >= 4:
                # print(cnt)
                temp_properties[c + 1] += 1
                temp_properties[c - 1] += \
                    100 / (temp_properties[2] + temp_properties[3])
                cnt -= 1
        temp_state[l, i, j] = c
    return temp_properties


def bool3d_to_onezero3d(arr: "np.ndarray") -> "np.ndarray":
    a, b, c = arr.shape  # a, b, c are three dimensions
    assert (a, b, c) == State.STATE_SHAPE
    result = np.zeros(a * b * c).reshape(a, b, c)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if arr[i][j][k]:
                    result[i][j][k] = 1
    return result


class NODE:
    def __init__(self, board: ArrayLike, hands: int, move, properties):
        """

        :param board: Board 6*6*6
        :param hands: AKA turns. From 0. Odd is white's turn, and even is black's turn.
        :param properties: [black score, white score, number of black lines, number of white lines]
        """
        self.board = board
        self.hands = hands
        self.move = move
        self.visiting_count = 0
        self.value_sum = 0
        self.is_leaf = True
        self.is_root = False
        self.children = []
        self.parent = []
        self.properties = properties

    def select(self):
        # search the tree, select a leaf node
        if self.is_leaf:
            return self
        else:
            p = -1
            next_node = self
            for child in self.children:
                temp_p = ucb(child.visiting_count, child.value_sum)
                if temp_p > p:
                    p = temp_p
                    next_node = child

            return next_node.select()

    def expand(self):
        # call the function which will return the legal moves
        # do the legal moves and add children to the leaf node
        self.is_leaf = False
        '''temp_board = np.zeros([6, 6, 6])
        for i in range(6):
            for j in range(6):
                for i in range(6):
                    temp_board[i][j][k] = self.board[i][j][k]'''
        temp_board = deepcopy(self.board)
        legal_moves = get_next_possible_move(temp_board)
        if self.hands % 2 == 0:
            color = 1  # black
        else:
            color = 2  # white
        for one_move in legal_moves:
            '''temp_board = np.zeros([6, 6, 6])
            for i in range(6):
                for j in range(6):
                    for i in range(6):
                        temp_board[i][j][k] = self.board[i][j][k]'''
            temp_board = deepcopy(self.board)
            '''temp_properties = [0, 0, 0, 0]
            for i in range(4):
                temp_properties[i] = self.properties[i]'''
            temp_properties = deepcopy(self.properties)
            temp_move = [0, 0, 0, 0]
            for i in range(3):
                temp_move[i] = one_move[i]
            temp_move[3] = color
            movements = [temp_move]
            new_properties = get_state_properties_b(temp_board, temp_properties, movements)
            temp_board[one_move[0]][one_move[1]][one_move[2]] = color
            new_child = NODE(temp_board, self.hands + 1, one_move, new_properties)
            self.children.append(new_child)
            new_child.parent = self

    def simulate(self):
        '''temp_board = np.zeros([6, 6, 6])
        start_board = np.zeros([6, 6, 6])
        for i in range(6):
            for j in range(6):
                for i in range(6):
                    temp_board[i][j][k] = self.board[i][j][k]
                    start_board[i][j][k] = self.board[i][j][k]'''
        temp_board = deepcopy(self.board)
        start_board = deepcopy(self.board)

        '''temp_properties = [0, 0, 0, 0]
        for i in range(4):
            temp_properties[i] = self.properties[i]'''
        temp_properties = deepcopy(self.properties)

        movements = []
        for n in range(self.hands, 64):
            legal_moves = get_next_possible_move(temp_board)
            m = random.randint(0, len(legal_moves) - 1)
            temp_move = [0, 0, 0, 0]
            for i in range(3):
                temp_move[i] = legal_moves[m][i]
            if n % 2 == 0:
                temp_move[3] = 1
                movements.append(temp_move)
                temp_board[temp_move[0]][temp_move[1]][temp_move[2]] = 1
            else:
                temp_move[3] = 2
                movements.append(temp_move)
                temp_board[temp_move[0]][temp_move[1]][temp_move[2]] = 2

        points = get_state_properties_b(start_board, temp_properties, movements)
        # black
        if self.hands % 2 == 0:
            if points[0] > points[1]:
                return 1
            else:
                return 0
        else:
            if points[1] > points[0]:
                return 1
            else:
                return 0

    def backup(self, reward):
        # from self to root
        # update visiting count and winning count
        global n
        n += 1
        temp_node = self
        flag = True
        if reward == 1:
            while not temp_node.is_root:
                if flag:
                    temp_node.value_sum += reward
                    flag = False
                else:
                    flag = True
                temp_node.visiting_count += 1
                temp_node = temp_node.parent
        else:
            while not temp_node.is_root:
                if flag:
                    flag = False
                else:
                    temp_node.value_sum += 1
                    flag = True
                temp_node.visiting_count += 1
                temp_node = temp_node.parent

    def play(self):
        pass

    def observation_tensors(self) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Get the tensors for input the neural network.
        Two kinds of tensors: 3D and scalar-vector.
            - 3D: Features from 3D state
                    Number of channels: 3
                        1. Black stones positions: fill 1 if it's a black piece's position else 0
                        2. White stones positions: fill 1 if it's a white piece's position else 0
                        3. Empty stones positions: fill 1 if it's a empty position else 0
                        4. Fill the entire channel with 0 if current player plays black else 1
            - scalar-vector: Scalar features. E.g. number of scores and lines
                    Number of channels: 4
                        1. Black player's score / 100
                        2. White player's score / 100
                        3. Number of black player's lines / 2
                        4. Number of black player's lines / 2
        :return: A tuple of two tensors
        """

        # 3D Features
        features_3d = [bool3d_to_onezero3d(np.equal(Color.BLACK, self.board)),
                       bool3d_to_onezero3d(np.equal(Color.WHITE, self.board)),
                       bool3d_to_onezero3d(np.equal(Color.EMPTY, self.board)),
                       bool3d_to_onezero3d(np.full(State.STATE_SHAPE, 0 if self.hands % 2 == 0 else 1))]
        features_3d = torch.tensor(features_3d)

        # Scalar Features
        features_scalar = [self.properties[0] / 100,
                           self.properties[1] / 100,
                           self.properties[2] / 2,
                           self.properties[3] / 2]
        features_scalar = torch.tensor(features_scalar)

        return features_3d, features_scalar


def mcts(root):
    root.is_root = True
    start = time.time()
    while True:
        temp_node = root.select()
        temp_node.expand()
        reward = temp_node.simulate()
        temp_node.backup(reward)
        end = time.time()
        if end - start >= 3:
            break

    temp_move = [0, 0, 0]
    value = 0
    for i in root.children:
        if i.visiting_count == 0:
            continue
        temp_value = i.value_sum / i.visiting_count
        if temp_value > value:
            value = temp_value
            temp_move = i.move

    print(n)
    return temp_move


class BasicBlock(nn.Module, ABC):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        x = x + y
        x = self.relu2(x)
        return x


class ResNet(nn.Module, ABC):
    def __init__(self, features3d_in_channels, blocks, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(features3d_in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            BasicBlock(in_channels=out_channels,
                       out_channels=out_channels) for _ in range(blocks)
        ])

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.convs:
            x = conv(x)
        return x


class NN3DConnect4(nn.Module, ABC):
    def __init__(self,
                 features_3d_in_channels: int,
                 features_scalar_in_channels: int,
                 channels: int,
                 blocks: int
                 ):
        super().__init__()

        self.features_3d_in_channels = features_3d_in_channels
        self.features_scalar_in_channels = features_scalar_in_channels
        self.channels = channels
        self.blocks = blocks
        self.num_players = 2
        self.num_distinct_actions = State.HEIGHT * State.WIDTH

        # Inputs
        self.features_3d_input_conv = nn.Conv3d(in_channels=features_3d_in_channels,
                                                out_channels=self.channels,
                                                kernel_size=5,
                                                stride=1,
                                                padding=2)
        self.features_scalar_input_fc = nn.Linear(in_features=features_scalar_in_channels,
                                                  out_features=self.channels)

        # Trunk
        self.trunk = ResNet(self.channels, self.blocks, self.channels)

        # Policy head
        self.policy_head_front = nn.Sequential(
            nn.Conv3d(in_channels=self.channels,
                      out_channels=2,
                      kernel_size=1),
            nn.BatchNorm3d(num_features=2),
            nn.ReLU(),
        )

        self.policy_head_end = nn.Sequential(
            nn.Linear(in_features=2 * State.DEPTH * State.HEIGHT * State.WIDTH,
                      out_features=self.num_distinct_actions),
            nn.Softmax(dim=1)
        )
        # TODO: filter the illegal move and re-normalize the probabilities

        # Value head
        self.value_head_front = nn.Sequential(
            nn.Conv3d(in_channels=self.channels,
                      out_channels=1,
                      kernel_size=1),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU(),
        )

        self.value_head_end = nn.Sequential(
            nn.Linear(in_features=State.DEPTH * State.HEIGHT * State.WIDTH,
                      out_features=self.channels),
            nn.ReLU(),
            nn.Linear(in_features=self.channels,
                      out_features=self.num_players),
            nn.Tanh()
        )

    def forward(self, x_3d, x_scalar):
        # Inputs
        x_3d = self.features_3d_input_conv(x_3d)
        x_scalar = self.features_scalar_input_fc(x_scalar)
        x_scalar = x_scalar.reshape(x_scalar.size() + (1, 1, 1))
        x = x_3d + x_scalar
        # Trunk
        x = self.trunk(x)
        # Policy head
        p = self.policy_head_front(x)
        p = p.view(-1, 2 * State.DEPTH * State.HEIGHT * State.WIDTH)
        p = self.policy_head_end(p)
        # Value head
        v = self.value_head_front(x)
        v = v.view(-1, State.DEPTH * State.HEIGHT * State.WIDTH)
        v = self.value_head_end(v)
        return p, v


if __name__ == '__main__':
    np1 = np.zeros([6, 6, 6])
    for k in range(0, 6):
        np1[k][0][0] = -1
        np1[k][0][1] = -1
        np1[k][0][4] = -1
        np1[k][0][5] = -1
        np1[k][1][0] = -1
        np1[k][1][5] = -1
        np1[k][4][0] = -1
        np1[k][4][5] = -1
        np1[k][5][0] = -1
        np1[k][5][1] = -1
        np1[k][5][4] = -1
        np1[k][5][5] = -1
    start_properties = [0, 0, 0, 0]
    a = NODE(np1, 0, None, start_properties)
    tensors = a.observation_tensors()
    # move = mcts(a)
    # print(move)

    #########
    model = NN3DConnect4(features_3d_in_channels=4,
                         features_scalar_in_channels=4,
                         channels=16,
                         blocks=4)
    model.eval()
    siz = 1
    for x in (4, 6, 6, 6):
        siz *= x
    t = torch.arange(float(siz)).reshape((1,) + (4, 6, 6, 6))
    print(t.size())

    st = time.time()
    n = 1000
    for i in range(800):
        model(torch.unsqueeze(tensors[0], 0).float(), torch.unsqueeze(tensors[1], 0).float())
    print(time.time() - st)
    ############3
