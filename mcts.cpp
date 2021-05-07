#include<iostream>
#include<vector>
#include<math.h>
#include<stdio.h>
#include<time.h>

using namespace std;

double ucb(int child_visiting_count,
    int winning_count,
    double parent_visit_cnt) {
    if (child_visiting_count == 0)
        child_visiting_count = 1;
    if (parent_visit_cnt == 0)
        parent_visit_cnt = 1;
    double p;
    p = log10((double)parent_visit_cnt) / (double)child_visiting_count;
    p = pow(p, 0.5) * 1.414;
    p = p + (double)winning_count / (double)child_visiting_count;
    return p;

}

class movement {
public:
    int l;
    int x;
    int y;
    int color;
    movement() {}
    movement(int l, int x, int y, int color) {
        this->l = l;
        this->x = x;
        this->y = y;
        this->color = color;
    }
};


class properties {
public:
    double black_points;
    double white_points;
    double black_lines;
    double white_lines;
    properties() {}
    properties(double black_points, 
               double white_points, 
               double black_lines, 
               double white_lines) {
        this->black_points = black_points;
        this->white_points = white_points;
        this->black_lines = black_lines;
        this->white_lines = white_lines;
    }
};

vector<movement> get_next_possible_move(int state[6][6][6]) {
    vector<movement> ret;
    ret.clear();
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (state[5][i][j] != 0)
                continue;
            for (int l = 0; l < 6; l++) {
                if (state[l][i][j] == 0) {
                    movement move(l, i, j, 0);
                    ret.push_back(move);
                    break;
                }
            }
        }
    }
    return ret;
}


bool boundary_test(int coordinate[3]) {
    for (int i = 0; i < 3; i++)
        if (coordinate[i] < 0 || coordinate[i] >= 6)
            return false;
    return true;
}


properties get_state_properties_b(int start_state[6][6][6], 
                               properties start_state_properties,
                               vector<movement> movements) {
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
    for (int l = 0; l < 6; l++) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                temp_state[l][i][j] = start_state[l][i][j];
            }
        }
    }
    properties temp_properties;
    temp_properties.black_points = start_state_properties.black_points;
    temp_properties.white_points = start_state_properties.white_points;
    temp_properties.black_lines = start_state_properties.black_lines;
    temp_properties.white_lines = start_state_properties.white_lines;

    for (int move_id = 0; move_id < movements.size(); move_id++) {
        int l = movements[move_id].l;
        int i = movements[move_id].x;
        int j = movements[move_id].y;
        int c = movements[move_id].color;
        for (int dir_id = 0; dir_id < 13; dir_id++) {
            int cnt = 1;
            for (int mul = 1; mul < 3; mul++) {
                int* temp_coordinate = new int[3];
                temp_coordinate[0] = l + dirs[dir_id][0] * mul;
                temp_coordinate[1] = i + dirs[dir_id][1] * mul;
                temp_coordinate[2] = j + dirs[dir_id][2] * mul;
                if (!boundary_test(temp_coordinate))
                    break;
                if (temp_state[temp_coordinate[0]][temp_coordinate[1]][temp_coordinate[2]] != c)
                    break;
                cnt += 1;
            }
            for (int mul = 1; mul < 3; mul++) {
                int* temp_coordinate = new int[3];
                temp_coordinate[0] = l + dirs[dir_id][0] * -mul;
                temp_coordinate[1] = i + dirs[dir_id][1] * -mul;
                temp_coordinate[2] = j + dirs[dir_id][2] * -mul;
                if (!boundary_test(temp_coordinate))
                    break;
                if (temp_state[temp_coordinate[0]][temp_coordinate[1]][temp_coordinate[2]] != c)
                    break;
                cnt += 1;
            }
            while (cnt >= 4) {
                if (c == 1) {
                    temp_properties.black_lines += 1;
                    temp_properties.black_points += 100 / (temp_properties.black_lines + temp_properties.white_lines);
                }
                else {
                    temp_properties.white_lines += 1;
                    temp_properties.white_points += 100 / (temp_properties.black_lines + temp_properties.white_lines);
                }
                cnt -= 1;
            }
        }
        temp_state[l][i][j] = c;
    }
    return temp_properties;
}

class Node {
public:
    int board[6][6][6];
    int hands;
    movement move;
    int visiting_count;
    int value_sum;
    bool is_leaf;
    bool is_root;
    vector<Node*> children;
    properties my_properties;
    Node* parent;


    Node(int board[6][6][6], 
        int hands,
        movement move,
        properties my_properties) {

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    this->board[i][j][k] = board[i][j][k];
                }
            }
        }
        this->hands = hands;
        this->move = move;
        this->my_properties = my_properties;
        this->visiting_count = 0;
        this->value_sum = 0;
        this->is_leaf = true;
        this->is_root = false;
        children.clear();

    }


    Node* select() {
        Node* cur = this;
        while (1) {
            if (cur->is_leaf)
                return cur;
            else {
                double p = -1;
                Node* next_node = cur;
                for (int i = 0; i < cur->children.size(); i++) {
                    Node* temp_node = cur->children[i];
                    double temp_p = ucb(temp_node->visiting_count, temp_node->value_sum, cur->visiting_count);
                    if (temp_p > p) {
                        p = temp_p;
                        next_node = temp_node;
                    }
                }
                cur = next_node;
            }
        }
    }


    void expand() {
        this->is_leaf = false;
        int temp_board[6][6][6];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    temp_board[i][j][k] = this->board[i][j][k];
                }
            }
        }
        vector<movement> legal_moves = get_next_possible_move(temp_board);
        int color;
        if (this->hands % 2 == 0)
            color = 1;
        else
            color = 2;
        for (int n = 0; n < legal_moves.size(); n++) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    for (int k = 0; k < 6; k++) {
                        temp_board[i][j][k] = this->board[i][j][k];
                    }
                }
            }
            movement temp_move;
            temp_move = legal_moves[n];
            temp_move.color = color;
            vector<movement> movements;
            movements.clear();
            movements.push_back(temp_move);
            properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
            temp_board[temp_move.l][temp_move.x][temp_move.y] = color;
            Node* new_child;
            new_child = new Node(temp_board, this->hands + 1, temp_move, new_properties);
            this->children.push_back(new_child);
            new_child->parent = this;

        }
    }


    int playout(properties start_properties, int hands) {
        int temp_board[6][6][6];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    temp_board[i][j][k] = this->board[i][j][k];
                }
            }
        }
        vector<movement> movements;
        movements.clear();
        for (int i = hands; i < 64; i++) {
            vector<movement> legal_moves = get_next_possible_move(temp_board);
            int random_number = rand() % legal_moves.size();
            movement temp_move = legal_moves[random_number];
            if (i % 2 == 0) {
                temp_move.color = 1;
                temp_board[temp_move.l][temp_move.x][temp_move.y] = 1;
            }
            else {
                temp_move.color = 2;
                temp_board[temp_move.l][temp_move.x][temp_move.y] = 2;
            }
            movements.push_back(temp_move);
        }
        properties end_properties = get_state_properties_b(board, start_properties, movements);
    
        if (hands % 2 == 0) {
            if (end_properties.black_points > end_properties.white_points)
                return 1;
            else
                return 0;
        }
        else {
            if (end_properties.white_points > end_properties.black_points)
                return 1;
            else
                return 0;
        }
    }


    void backup(int reward) {
        Node* cur = this;
        bool flag = true;
        if (reward == 1) {
            while (1) {
                if (flag) {
                    cur->value_sum += reward;
                    flag = false;
                }
                else {
                    flag = true;
                }
                cur->visiting_count ++;
                if (cur->is_root)
                    break;
                cur = cur->parent;
            }
        }
        else {
            while (1) {
                if (flag)
                    flag = false;
                else {
                    cur->value_sum += 1;
                    flag = true;
                       
                }
                cur->visiting_count += 1;
                if (cur->is_root)
                    break;
                cur = cur->parent;
            }
        }
    }
};

class MCTS {
public:
    Node* root;
    int cur_simulation_cnt;
    int max_simulation_cnt;
    int max_time_sec;


    MCTS(Node* root, int max_simulation_cnt, int max_time_sec) {
        this->root = root;
        this->max_simulation_cnt = max_simulation_cnt;
        this->max_time_sec = max_time_sec;
        this->cur_simulation_cnt = 0;
    }


    movement run() {
        this->root->is_root = true;
        clock_t start = clock();
        while (1) {
            //cout << 1 << endl;
            Node* temp_node = this->root->select();
            //cout << 2 << endl;
            temp_node->expand();
            //cout << 3 << endl;
            int reward = temp_node->playout(temp_node->my_properties, temp_node->hands);
            //cout << 4 << endl;
            temp_node->backup(reward);
            this->cur_simulation_cnt++;
            clock_t end = clock();
            if (end - start >= this->max_time_sec * 1000)
                break;
            if (this->cur_simulation_cnt >= this->max_simulation_cnt)
                break;
        }
        double winning_rate = 0;
        movement ret;
        for (int i = 0; i < this->root->children.size(); i++) {
            Node* temp_node = this->root->children[i];
            if (temp_node->visiting_count == 0)
                continue;
            //cout << temp_node->value_sum << " " << temp_node->visiting_count << endl;
            double temp_winning_rate = (double)temp_node->value_sum / (double)temp_node->visiting_count;
            //cout << temp_winning_rate << endl;
            if (temp_winning_rate > winning_rate) {
                winning_rate = temp_winning_rate;
                ret = temp_node->move;
                //cout << temp_node->move.l << " " << temp_node->move.x << " " << temp_node->move.y << endl;
            }
        }
        cout << "Simulation cnt: " << this->cur_simulation_cnt << endl;
        return ret;
    }
};

int main() {

    int b[6][6][6];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                b[i][j][k] = 0;
            }
        }
    }
    for (int k = 0; k < 6; k++) {
        b[k][0][0] = -1;
        b[k][0][1] = -1;
        b[k][0][4] = -1;
        b[k][0][5] = -1;
        b[k][1][0] = -1;
        b[k][1][5] = -1;
        b[k][4][0] = -1;
        b[k][4][5] = -1;
        b[k][5][0] = -1;
        b[k][5][1] = -1;
        b[k][5][4] = -1;
        b[k][5][5] = -1;
    }
       
    properties start_properties(0, 0, 0, 0);
    movement move, next_move;
    Node* start_node = new Node(b, 0, move, start_properties);
    MCTS mcts(start_node, 9999, 3);
    next_move = mcts.run();
    cout << next_move.l << " " << next_move.x << " " << next_move.y << endl;
}