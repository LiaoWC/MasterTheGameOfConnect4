import numpy as np
import random
import math
import time
from numpy.typing import ArrayLike
from typing import List, Sequence, Tuple, Optional, Union
from copy import deepcopy
import torch
from abc import ABC
from torch import nn
from math import sqrt
import argparse
import pathlib
import yaml

# Hyper-parameters
UCB_C = 1.414
C_PUCT = 1.5
C_POLICY_TARGET_PRUNING = 2


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


def ucb(child_visiting_count, winning_count, parent_visit_cnt):
    if child_visiting_count == 0:
        child_visiting_count = 1.
    p = math.log(1. if parent_visit_cnt == 0 else float(parent_visit_cnt), 10) / child_visiting_count
    p = pow(p, 0.5) * UCB_C
    p = p + winning_count / child_visiting_count
    return p


def puct(value_sum: float, visit_cnt: int, child_p: float, child_visit_cnt: int):
    # assert visit_cnt > 0
    q = value_sum / float(1 if visit_cnt == 0 else visit_cnt)
    u = C_PUCT * child_p * sqrt(1. if visit_cnt == 0 else float(visit_cnt)) / float(1 + child_visit_cnt)
    return q + u


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


class EvalFunc:
    # TODO: Ensure all eval function's all the input is one that has been deep-copied
    """
    # TODO: Ensure all eval function's input are in the same format: board, properties, hands, **kwargs.
    """

    @staticmethod
    def playout(board, properties, hands) -> float:
        # temp_board = deepcopy(self.board)
        # start_board = deepcopy(self.board)

        # temp_properties = deepcopy(self.properties)
        temp_board = board
        start_board = deepcopy(board)

        movements = []
        for n in range(hands, 64):
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

        points = get_state_properties_b(start_board, properties, movements)
        # black
        if hands % 2 == 0:
            if points[0] > points[1]:
                return 1
            else:
                return 0
        else:
            if points[1] > points[0]:
                return 1
            else:
                return 0

    @staticmethod
    def nn(board, properties, hands, model, device) -> (np.ndarray, float):
        tensors = MCTS.observation_tensors(board, properties, hands)
        # Turn to batch with batch size 1
        tensor_3d, tensor_scalar = torch.unsqueeze(tensors[0], 0).float(), torch.unsqueeze(tensors[1], 0).float()
        # To device
        tensor_3d = tensor_3d.to(device)
        tensor_scalar = tensor_scalar.to(device)
        model = model.to(device)
        # Get p and v
        with torch.no_grad():
            p, v = model.forward(tensor_3d, tensor_scalar)
        p = p.cpu().numpy()[0]
        v = v.cpu().tolist()[0][0]
        return p, v


# TODO: Batch size > 1 when inference?


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
        self.children: List["NODE"] = []
        self.parent = []
        self.properties = properties
        self.child_probs: List[float] = []

    def select(self, select_func: str):
        # search the tree, select a leaf node
        if self.is_leaf:
            return self
        else:
            p = -1
            next_node = self
            for i, child in enumerate(self.children):
                if select_func == 'ucb':
                    temp_p = ucb(child.visiting_count, child.value_sum, self.visiting_count)
                elif select_func == 'puct':
                    temp_p = puct(value_sum=self.value_sum, visit_cnt=self.visiting_count,
                                  child_p=self.child_probs[i], child_visit_cnt=self.children[i].visiting_count)
                else:
                    raise ValueError('Invalid select_func: {}'.format(select_func))
                if temp_p > p:
                    p = temp_p
                    next_node = child
            return next_node.select(select_func)

    def expand(self):
        # call the function which will return the legal moves
        # do the legal moves and add children to the leaf node
        self.is_leaf = False
        # '''temp_board = np.zeros([6, 6, 6])
        # for i in range(6):
        #     for j in range(6):
        #         for i in range(6):
        #             temp_board[i][j][k] = self.board[i][j][k]'''
        temp_board = deepcopy(self.board)
        legal_moves = get_next_possible_move(temp_board)
        if self.hands % 2 == 0:
            color = 1  # black
        else:
            color = 2  # white
        for one_move in legal_moves:
            # '''temp_board = np.zeros([6, 6, 6])
            # for i in range(6):
            #     for j in range(6):
            #         for i in range(6):
            #             temp_board[i][j][k] = self.board[i][j][k]'''
            temp_board = deepcopy(self.board)
            # '''temp_properties = [0, 0, 0, 0]
            # for i in range(4):
            #     temp_properties[i] = self.properties[i]'''
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

    def backup(self, reward):
        # from self to root
        # update visiting count and winning count
        temp_node = self
        flag = True
        if reward == 1:
            while True:
                if flag:
                    temp_node.value_sum += reward
                    flag = False
                else:
                    flag = True
                temp_node.visiting_count += 1
                if temp_node.is_root:
                    break
                temp_node = temp_node.parent
        else:
            while True:
                if flag:
                    flag = False
                else:
                    temp_node.value_sum += 1
                    flag = True
                temp_node.visiting_count += 1
                if temp_node.is_root:
                    break
                temp_node = temp_node.parent

    def get_node_after_playing(self, move):
        temp_board = deepcopy(self.board)
        temp_properties = deepcopy(self.properties)
        if self.hands % 2 == 0:
            color = 1
        else:
            color = 2
        temp_move = [0, 0, 0, 0]
        for i in range(3):
            temp_move[i] = move[i]
        temp_move[3] = color
        movements = [temp_move]
        new_properties = get_state_properties_b(temp_board, temp_properties, movements)
        temp_board[move[0]][move[1]][move[2]] = color
        return NODE(board=temp_board,
                    hands=self.hands + 1,
                    move=move,
                    properties=new_properties)


class MCTS:
    def __init__(self,
                 root_node: "NODE",
                 max_time_sec: int,
                 eval_func: str,
                 model=None,
                 device=None,
                 max_simulation_cnt: int = 999999,
                 not_greedy: bool = False,
                 dirichlet_noise: bool = False,
                 dirichlet_epsilon: float = 0.25,
                 dirichlet_alpha: float = 0.05):
        """
        Available evaluation functions current:
            1. playout
            2. nn:
                    - Need provide model, and device.

        :param root_node:
        :param max_simulation_cnt:
        :param eval_func:
        :param model:
        :param not_greedy: Use prob to pick up a move rather than pick up best-confident child. This only affect \
            when you use nn to eval.
        :param dirichlet_noise: Use Dirichlet noise or not. Only affect nn.
        """
        self.root = root_node
        self.cur_simulation_cnt = 0
        self.max_simulation_cnt = max_simulation_cnt
        self.max_time_sec = max_time_sec
        self.eval_func = eval_func
        self.model = model
        self.device = device
        self.not_greedy = not_greedy
        self.dirichlet_noise = dirichlet_noise
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        # Check evaluation function
        if eval_func not in ['playout', 'nn']:
            raise ValueError('Invalid eval_func: {}'.format(eval_func))
        if eval_func == 'nn':
            if not model or not device:
                raise ValueError('You must provide the parameter "model" and "device" when the eval_func is "nn".')
            self.model.to(device)

    def run(self, return_simulation_cnt=False, return_time_used=False):
        self.root.is_root = True
        start = time.time()
        while True:
            #############
            # Select
            ##############
            if self.eval_func == 'playout':
                select_func = 'ucb'
            elif self.eval_func == 'nn':
                select_func = 'puct'
            else:
                raise ValueError('Invalid eval_func: {}'.format(self.eval_func))
            temp_node = self.root.select(select_func)
            ########
            # Expand
            ########
            temp_node.expand()
            ################################
            # Eval
            ################################
            board, properties, hands = deepcopy(temp_node.board), deepcopy(temp_node.properties), deepcopy(
                temp_node.hands)
            if self.eval_func == 'playout':
                reward = EvalFunc.playout(board, properties, hands)
            elif self.eval_func == 'nn':
                ######################################################################
                p, v = EvalFunc.nn(board, properties, hands, self.model, self.device)
                # Do dirichlet on root's child probs
                dirichlet_samples = []
                dirichlet_sum = 0.
                if temp_node.is_root and self.dirichlet_noise:
                    dirichlet_samples = np.random.dirichlet(np.full((len(temp_node.children),),
                                                                    self.dirichlet_alpha)).tolist()
                # Reshape p to be 2D
                p = p.reshape(State.HEIGHT, State.WIDTH)
                # Filter out illegal moves and re-normalize the probability
                if len(temp_node.children) == 0:
                    raise ValueError('You must this this function after the node is expanded.')
                for idx, child in enumerate(temp_node.children):
                    child_prob = p[child.move[1]][child.move[2]]
                    if temp_node.is_root and self.dirichlet_noise:
                        child_prob = (1 - self.dirichlet_epsilon) * child_prob + self.dirichlet_epsilon * \
                                     dirichlet_samples[idx]
                    temp_node.child_probs.append(child_prob)
                probs_sum = sum(temp_node.child_probs)
                for i in range(len(temp_node.children)):
                    temp_node.child_probs[i] = temp_node.child_probs[i] / probs_sum
                reward = v
                ######################################################################
            else:
                raise ValueError('Invalid eval_func: {}'.format(self.eval_func))
            ############
            # Backup
            ############
            temp_node.backup(reward)
            self.cur_simulation_cnt += 1
            end = time.time()
            time_used = end - start
            ####################
            # Check time limit
            ####################
            if end - start >= self.max_time_sec:
                break
            ################################
            # Check simulation count limit
            ################################
            if self.cur_simulation_cnt >= self.max_simulation_cnt:
                break

        #############################################
        # After each simulation, pick a move to play
        #############################################
        if self.eval_func == 'playout':
            picked_move = [0, 0, 0]
            value = 0
            for i in self.root.children:
                if i.visiting_count == 0:
                    continue
                temp_value = i.value_sum / i.visiting_count
                if temp_value > value:
                    value = temp_value
                    picked_move = i.move

        elif self.eval_func == 'nn':
            distribution = self.get_root_child_distribution(normalize=True)
            if self.not_greedy:
                picked_idx = random.choices(range(0, len(distribution)), distribution)[0]
            else:
                max_value = max(distribution)
                max_value_indices = [x for i, x in enumerate(distribution) if x == max_value]
                picked_idx = random.randint(0, len(max_value_indices) - 1)
            picked_move = self.root.children[picked_idx].move
        else:
            raise ValueError('Invalid eval_func here: {}'.format(self.eval_func))
        ##########
        # Return
        ##########
        rtn_list = [picked_move]
        if return_simulation_cnt:
            rtn_list.append(self.cur_simulation_cnt)
        if return_time_used:
            rtn_list.append(time_used)
        return tuple(rtn_list)

    @staticmethod
    def observation_tensors(board, properties, hands) -> Tuple["torch.Tensor", "torch.Tensor"]:
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
        features_3d = [bool3d_to_onezero3d(np.equal(Color.BLACK, board)),
                       bool3d_to_onezero3d(np.equal(Color.WHITE, board)),
                       bool3d_to_onezero3d(np.equal(Color.EMPTY, board)),
                       bool3d_to_onezero3d(np.full(State.STATE_SHAPE, 0 if hands % 2 == 0 else 1))]
        features_3d = torch.tensor(features_3d)

        # Scalar Features
        features_scalar = [properties[0] / 100,
                           properties[1] / 100,
                           properties[2] / 2,
                           properties[3] / 2]
        features_scalar = torch.tensor(features_scalar)

        return features_3d, features_scalar

    def get_root_child_distribution(self, normalize: bool) -> Union[List[int], List[float]]:
        if self.root.visiting_count <= 1:
            raise ValueError("Root's visit cnt must > 2 when you use this function.")
        distribution = [child.visiting_count for child in self.root.children]
        if not normalize:
            return distribution
        else:
            # Normalize to get proportions whose sum is 1.
            distribution_sum = sum(distribution)
            return [cnt / float(distribution_sum) for cnt in distribution]

    def get_root_child_distribution_2d(self, normalize: bool) -> np.ndarray:
        distribution = self.get_root_child_distribution(normalize=normalize)
        distribution_2d = np.zeros(State.HEIGHT * State.WIDTH).reshape(State.HEIGHT, State.WIDTH)
        for i, child in enumerate(self.root.children):
            distribution_2d[child.move[1]][child.move[2]] = distribution[i]
        return distribution_2d

    @staticmethod
    def get_init_node():
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
        return NODE(np1, 0, None, start_properties)


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
                      out_features=1),
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
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', help='Path to config file.', required=True)
    args = parser.parse_args()

    # Read config from training_config.yaml
    config = yaml.safe_load(pathlib.Path(args.config).read_text())

    ###########################
    # How to do "playout" mcts?
    ###########################
    # root = MCTS.get_init_node()
    # mcts = MCTS(root,
    #             max_simulation_cnt=9999,
    #             eval_func='playout', max_time_sec=3)
    # move = mcts.run()
    # print(move)
    ########################
    # How to use nn?
    ########################
    root = MCTS.get_init_node()
    # Load neural network model
    net = NN3DConnect4(features_3d_in_channels=4,
                       features_scalar_in_channels=4,
                       channels=config['model']['channels'],
                       blocks=config['model']['blocks'])
    ########################
    # How to load nn model?
    ########################
    # model_path = '????'
    # net.load_state_dict(torch.load(model_path))
    #######################
    # How to use MCTS?
    #######################
    mcts = MCTS(root, max_simulation_cnt=9999, eval_func='nn', model=net,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                max_time_sec=3, not_greedy=False, dirichlet_noise=False)
    print(mcts.run(return_simulation_cnt=True))
    mcts = MCTS(root, max_simulation_cnt=9999, eval_func='nn', model=net,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                max_time_sec=3, not_greedy=False, dirichlet_noise=True)
    print(mcts.run(return_simulation_cnt=True))
