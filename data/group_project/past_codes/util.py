import numpy as np


def get_state_properties_a(pre_state, pre_state_properties, now_state, is_black):
    temp_state = now_state - pre_state
    movements = np.zeros([2, 4])
    for l in range(6):
        for i in range(6):
            for j in range(6):
                if temp_state[l][i][j] == 1:
                    movements[1 if is_black else 0, :] = (l, i, j, 1)
                elif temp_state[l][i][j] == 2:
                    movements[0 if is_black else 1, :] = (l, i, j, 2)
    ret = get_state_properties(pre_state, pre_state_properties, movements)
    return ret


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



def random_gen_board(steps):
    board = np.zeros([6, 6, 6])
    for l in range(6):
        board[l, 0, 0] = -1
        board[l, 0, 1] = -1
        board[l, 0, 4] = -1
        board[l, 0, 5] = -1
        board[l, 1, 0] = -1
        board[l, 1, 5] = -1
        board[l, 4, 0] = -1
        board[l, 4, 5] = -1
        board[l, 5, 0] = -1
        board[l, 5, 1] = -1
        board[l, 5, 4] = -1
        board[l, 5, 5] = -1
    movements = []
    for step_num in range(steps):
        possible_moves = get_next_possible_move(board)
        choose_index = np.random.randint(len(possible_moves))
        l, i, j = possible_moves[choose_index][0], possible_moves[choose_index][1], possible_moves[choose_index][2]
        c = step_num % 2 + 1
        movements.append([l, i, j, c])
        board[l, i, j] = c
    print(board)
    return movements

def gen_block_move(state, is_black):
    all_possible_moves = get_next_possible_move(state)
    opponent_color = 2 if is_black else 1
    dirs = []
    vec = [-1, 0, 1]
    for l in vec:
        for i in vec:
            for j in vec:
                if l == 0 and i == 0 and j == 0:
                    continue
                dirs.append([l, i, j])
    block_moves = []
    for move in all_possible_moves:
        for dir in dirs:
            new_pos = np.add(move, dir)
            
            l, i, j = new_pos[0], new_pos[1], new_pos[2]
            if not boundary_test(new_pos):
                continue
            if state[l, i, j] != opponent_color:
                continue
            new_pos = np.add(new_pos, dir)
            l, i, j = new_pos[0], new_pos[1], new_pos[2]
            if not boundary_test(new_pos):
                continue
            if state[l, i, j] != opponent_color:
                continue
            
            block_moves.append(move)
    if block_moves:
        choose_index = np.random.randint(len(block_moves))
        return block_moves[choose_index]
    else:
        choose_index = np.random.randint(len(all_possible_moves))
        return all_possible_moves[choose_index]


def get_rotate_and_mirror(board):
    b1 = np.zeros([6, 6, 6])
    b2 = np.zeros([6, 6, 6])
    b3 = np.zeros([6, 6, 6])
    b4 = np.zeros([6, 6, 6])
    b5 = np.zeros([6, 6, 6])
    b6 = np.zeros([6, 6, 6])
    b7 = np.zeros([6, 6, 6])
    b8 = np.zeros([6, 6, 6])
    for i in range(6):
        for j in range(6):
            for k in range(6):
                b1[i][j][k] = board[i][j][k]
                b2[i][k][5 - j] = board[i][j][k]
                b3[i][5 - j][5 - k] = board[i][j][k]
                b4[i][5 - k][j] = board[i][j][k]
                b5[i][j][5 - k] = board[i][j][k]
                b6[i][5 -k][5 - j] = board[i][j][k]
                b7[i][5 - j][k] = board[i][j][k]
                b8[i][k][j] = board[i][j][k]
    board_list = [b1, b2, b3, b4, b5, b6, b7, b8]
    ret = np.array(board_list)
    return ret