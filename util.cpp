#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <time.h>
#include <cstdio>
using namespace std;
int* get_state_properties_a(int*** pre_state, int*  pre_state_properties, int*** now_state, bool is_black);
bool boundary_test(int* coordinate);
int* get_state_properties_b(int*** start_state, int* start_state_properties, int** movements);
int** get_next_possible_move(int*** state);
int** random_gen_board(int steps);
int* gen_block_move(int*** state, bool is_black);
int**** get_rotate_and_mirror(int*** board);

int* get_state_properties_a(int*** pre_state, int*  pre_state_properties, int*** now_state, bool is_black){
    int temp_state[6][6][6];
    for(int l = 0; l < 6 ; l++){
        for(int i = 0; i < 6 ; i++){
            for(int j = 0; j < 6; j++){
                temp_state[l][i][j] = now_state[l][i][j] - pre_state[l][i][j];             
            }
        }
    }
    int** movements = new int* [2];
    movements[0] = new int [4];
    movements[1] = new int [4];
    for(int l = 0; l < 6 ; l++){
        for(int i = 0; i < 6 ; i++){
            for(int j = 0; j < 6; j++){
                if (temp_state[l][i][j] == 1){
                    int r = (is_black)?1:0;
                    movements[r][0] = l;
                    movements[r][1] = i;
                    movements[r][2] = j;
                    movements[r][3] = 1;
                }
                else if (temp_state[l][i][j] == 2){
                    int r = (is_black)?0:1;
                    movements[r][0] = l;
                    movements[r][1] = i;
                    movements[r][2] = j;
                    movements[r][3] = 1;
                }
            }
        }
    }
    int* ret = get_state_properties_b(pre_state, pre_state_properties, movements);
    return ret;
}


bool boundary_test(int* coordinate){
    for(int i = 0 ; i < 3 ; i++)
        if (coordinate[i] < 0 || coordinate[i] >= 6)
            return false;
    return true;
}


int* get_state_properties_b(int*** start_state, int* start_state_properties, int** movements){
    int dirs[13][3] = {
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0},
        {0, 1, -1},
        {1, 0, -1},
        {1, -1, 0},
        {1, 1, 1},
        {1, 1, -1},
        {1, -1, 1},
        {-1, 1, 1}
    };
    int temp_state[6][6][6];
    for(int l = 0; l < 6 ; l++){
        for(int i = 0; i < 6 ; i++){
            for(int j = 0; j < 6; j++){
                temp_state[l][i][j] = start_state[l][i][j];             
            }
        }
    }
    int* temp_properties = new int [4];
    for(int i = 0 ; i < 4; i++){
        temp_properties[i] = start_state_properties[i];
    }

    for(int move_id = 0; move_id < sizeof(movements)/sizeof(movements[0]); move_id++){
        int l = movements[move_id][0], i = movements[move_id][1], j = movements[move_id][2], c = movements[move_id][3];
        for (int dir_id = 0; dir_id < 13 ; dir_id++){
            int cnt = 1;
            for (int mul = 1 ; mul < 3 ; mul++){
                int* temp_coordinate = new int [3];
                temp_coordinate[0] = l + dirs[dir_id][0] * mul;
                temp_coordinate[1] = i + dirs[dir_id][1] * mul;
                temp_coordinate[2] = j + dirs[dir_id][2] * mul;
                if (!boundary_test(temp_coordinate))
                    break;
                if (temp_state[temp_coordinate[0]][temp_coordinate[1]][temp_coordinate[2]] != c)
                    break;
                cnt += 1;
            }
            for (int mul = 1 ; mul < 3 ; mul++){
                int* temp_coordinate = new int [3];
                temp_coordinate[0] = l + dirs[dir_id][0] * -mul;
                temp_coordinate[1] = i + dirs[dir_id][1] * -mul;
                temp_coordinate[2] = j + dirs[dir_id][2] * -mul;
                if (!boundary_test(temp_coordinate))
                    break;
                if (temp_state[temp_coordinate[0]][temp_coordinate[1]][temp_coordinate[2]] != c)
                    break;
                cnt += 1;
            }
            while (cnt >= 4){
                temp_properties[c + 1] += 1;
                temp_properties[c - 1] +=  \
                    100 / (temp_properties[2] + temp_properties[3]);
                cnt -= 1;
            }
        }
        temp_state[l][i][j] = c;
    }
    return temp_properties;
}



int** get_next_possible_move(int*** state){
    vector<int> v;
    for(int i = 0 ; i < 6 ; i++){
        for(int j = 0 ; j < 6 ; j++){
            if (state[5][i][j] != 0)
                continue;
            for(int l = 0 ; l < 6 ; l++){
                if (state[l][i][j] == 0){
                    v.push_back(l);
                    v.push_back(i);
                    v.push_back(j);
                    break;
                }
            }
        }
    }
    int size = v.size()/3;
    int** ret = new int* [size];
    for(int i = 0 ; i < size ; i++){
        for(int j = 0; j < 3 ; j++){
            ret[i][j] = v[3 * i + j];
        }
    }
    return ret;
}


int** random_gen_board(int steps){
    int*** board = new int** [6];
    for(int i = 0 ; i < 6 ; i++){
        board[i] = new int* [6];
        for(int j = 0; j < 6; j++){
            board[i][j] = new int [6];
        }
    }
    memset(board, 0, sizeof(board));
    for(int l = 0 ; l < 6 ; l++){
        board[l][0][0] = -1;
        board[l][0][1] = -1;
        board[l][0][4] = -1;
        board[l][0][5] = -1;
        board[l][1][0] = -1;
        board[l][1][5] = -1;
        board[l][4][0] = -1;
        board[l][4][5] = -1;
        board[l][5][0] = -1;
        board[l][5][1] = -1;
        board[l][5][4] = -1;
        board[l][5][5] = -1;
    }
    vector<int> v;
    srand(time(0));
    for(int step_num = 0 ; step_num < steps ; step_num++){
        int** possible_moves = get_next_possible_move(board);
        int choose_index = rand() % (sizeof(possible_moves)/sizeof(possible_moves[0]));
        int l = possible_moves[choose_index][0], i = possible_moves[choose_index][1], j = possible_moves[choose_index][2];
        int c = step_num % 2 + 1;
        v.push_back(l);
        v.push_back(i);
        v.push_back(j);
        v.push_back(c);
        board[l][i][j] = c;
    }
    int size = v.size()/4;
    int** ret = new int* [size];
    for(int i = 0 ; i < size ; i++){
        for(int j = 0; j < 4 ; j++){
            ret[i][j] = v[4 * i + j];
        }
    }
    return ret;
}

int* gen_block_move(int*** state, bool is_black){
    int** all_possible_moves = get_next_possible_move(state);
    int opponent_color = (is_black)?2:1;
    int dirs[26][3];
    int cnt = 0;
    for(int l = -1 ; l <= 1 ; l++){
        for(int i = -1 ; i <= 1 ; i++){
            for(int j = -1 ; j <= 1 ; j++){
                if (l == 0 && i == 0 && j == 0)
                    continue;
                dirs[cnt][0] = l;
                dirs[cnt][1] = i;
                dirs[cnt][2] = j;
                cnt++;
            }
        }
    }
    vector<int> block_moves;
    for(int move_num = 0 ; move_num < sizeof(all_possible_moves)/sizeof(all_possible_moves[0]) ; move_num){
        for(int dir_num = 0 ; dir_num < 26 ; dir_num++){
            int* new_pos = new int [3];
            new_pos[0] = all_possible_moves[move_num][0] + dirs[dir_num][0];
            new_pos[1] = all_possible_moves[move_num][1] + dirs[dir_num][1];
            new_pos[2] = all_possible_moves[move_num][2] + dirs[dir_num][2];

            int l = new_pos[0], i = new_pos[1], j = new_pos[2];
            if(!boundary_test(new_pos) || state[l][i][j] != opponent_color)
                continue;
            new_pos[0] += dirs[dir_num][0];
            new_pos[1] += dirs[dir_num][1];
            new_pos[2] += dirs[dir_num][2];
            l = new_pos[0]; i = new_pos[1]; j = new_pos[2];
            if(!boundary_test(new_pos) || state[l][i][j] != opponent_color)
                continue;
            
            block_moves.push_back(all_possible_moves[move_num][0]);
            block_moves.push_back(all_possible_moves[move_num][1]);
            block_moves.push_back(all_possible_moves[move_num][2]);
        }
    }
    srand(time(0));
    if (block_moves.size() != 0){
        int choose_index = rand() % (block_moves.size() / 3);
        int* ret = new int [3];
        for(int j = 0 ; j < 3 ; j++)
            ret[j] = block_moves[choose_index * 3 + j];
        return ret;
    }
    else{
        int choose_index = rand() % (sizeof(all_possible_moves)/sizeof(all_possible_moves[0]));
        return all_possible_moves[choose_index];
    }
}

int**** get_rotate_and_mirror(int*** board){
    int**** board_list = new int*** [8];
    memset(board_list, 0, sizeof(board_list));
    for(int l = 0 ; l < 6 ; l++){
        for(int i = 0 ; i < 6 ; i++){
            for(int j = 0 ; j < 6 ; j++){
                board_list[0][l][i][j] = board[l][i][j];
                board_list[1][l][j][5 - i] = board[l][i][j];
                board_list[2][l][5 - i][5 - j] = board[l][i][j];
                board_list[3][l][5 - j][i] = board[l][i][j];
                board_list[4][l][i][5 - j] = board[l][i][j];
                board_list[5][l][5 -j][5 - i] = board[l][i][j];
                board_list[6][l][5 - i][j] = board[l][i][j];
                board_list[7][l][j][i] = board[l][i][j];
            }
        }
    }
    return board_list; 
}

int main(){
    return 0;
}

