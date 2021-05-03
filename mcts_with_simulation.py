import numpy as np
import random
import math
import time

# the coefficient for ucb
c = 1.414

# global variable for counting all playing times
n = 0


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
                print(cnt)
                temp_properties[c + 1] += 1
                temp_properties[c - 1] +=  \
                    100 / (temp_properties[2] + temp_properties[3])
                cnt -= 1
        temp_state[l, i, j] = c
    return temp_properties


class NODE:
    def __init__(self, board, hands, move, properties):
        self.board = board
        self.hands = hands
        self.move = move
        self.visiting_count = 0
        self.winning_count = 0
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
                temp_p = ucb(child.visiting_count, child.winning_count)
                if temp_p > p:
                    p = temp_p
                    next_node = child

            return next_node.select()

    def expand(self):
        # call the function which will return the legal moves
        # do the legal moves and add children to the leaf node
        self.is_leaf = False
        temp_board = np.zeros([6, 6, 6])
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    temp_board[i][j][k] = self.board[i][j][k]
        legal_moves = get_next_possible_move(temp_board)
        if self.hands % 2 == 0:
            m = 1     # black
        else:
            m = 2     # white
        for one_move in legal_moves:
            temp_board = np.zeros([6, 6, 6])
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        temp_board[i][j][k] = self.board[i][j][k]

            temp_properties = [0, 0, 0, 0]
            for i in range(4):
                temp_properties[i] = self.properties[i]

            temp_move = [0, 0, 0, 0]
            for i in range(3):
                temp_move[i] = one_move[i]
            temp_move[3] = m
            movements = []
            movements.append(temp_move)
            new_properties = get_state_properties_b(temp_board, temp_properties, movements)
            temp_board[one_move[0]][one_move[1]][one_move[2]] = m
            new_child = NODE(temp_board, self.hands + 1, one_move, new_properties)
            self.children.append(new_child)
            new_child.parent = self

    def simulate(self):
        start_board = np.zeros([6, 6, 6])
        temp_board = np.zeros([6, 6, 6])
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    temp_board[i][j][k] = self.board[i][j][k]
                    start_board[i][j][k] = self.board[i][j][k]
        temp_properties = [0, 0, 0, 0]
        for i in range(4):
            temp_properties[i] = self.properties[i]

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
        while not temp_node.is_root:
            if flag:
                temp_node.winning_count += reward
                flag = False
            else:
                flag = True
            temp_node.visiting_count += 1
            temp_node = temp_node.parent

    def play(self):
        pass


def mcts(root):
    root.is_root = True
    count = 0
    start = time.time()
    while True:
        count += 1
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
        temp_value = i.winning_count / i.visiting_count
        if temp_value > value:
            value = temp_value
            temp_move = i.move

    print(count)
    return temp_move


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
    move = mcts(a)
    print(move)
