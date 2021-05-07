#include<iostream>
#include<vector>
#include<cmath>
#include<ctime>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> sec;

using namespace std;


double ucb(int child_visiting_count,
           int winning_count,
           double parent_visit_cnt) {
    if (child_visiting_count == 0)
        child_visiting_count = 1;
    if (parent_visit_cnt == 0)
        parent_visit_cnt = 1;
    double p;
    p = log10((double) parent_visit_cnt) / (double) child_visiting_count;
    p = pow(p, 0.5) * 1.414;
    p = p + (double) winning_count / (double) child_visiting_count;
    return p;

}

class Movement {
public:
    int l{};
    int x{};
    int y{};
    int color{};

    Movement() = default;

    Movement(int l, int x, int y, int color) {
        this->l = l;
        this->x = x;
        this->y = y;
        this->color = color;
    }
};


class Properties {
public:
    double black_points{};
    double white_points{};
    double black_lines{};
    double white_lines{};

    Properties() = default;

    Properties(double black_points,
               double white_points,
               double black_lines,
               double white_lines) {
        this->black_points = black_points;
        this->white_points = white_points;
        this->black_lines = black_lines;
        this->white_lines = white_lines;
    }
};

bool boundary_test(const int coordinate[3]) {
    for (int i = 0; i < 3; i++)
        if (coordinate[i] < 0 || coordinate[i] >= 6)
            return false;
    return true;
}


Properties get_state_properties_b(int start_state[6][6][6],
                                  Properties start_state_properties,
                                  const vector<Movement> &movements) {
    int dirs[13][3] = {
            {0,  0,  1},
            {0,  1,  0},
            {1,  0,  0},
            {0,  1,  1},
            {1,  0,  1},
            {1,  1,  0},
            {0,  1,  -1},
            {1,  0,  -1},
            {1,  -1, 0},
            {1,  1,  1},
            {1,  1,  -1},
            {1,  -1, 1},
            {-1, 1,  1}
    };
    int temp_state[6][6][6];
    for (int l = 0; l < 6; l++) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                temp_state[l][i][j] = start_state[l][i][j];
            }
        }
    }
    Properties temp_properties;
    temp_properties.black_points = start_state_properties.black_points;
    temp_properties.white_points = start_state_properties.white_points;
    temp_properties.black_lines = start_state_properties.black_lines;
    temp_properties.white_lines = start_state_properties.white_lines;

    for (auto &movement : movements) {
        int l = movement.l;
        int i = movement.x;
        int j = movement.y;
        int c = movement.color;
        for (auto &dir : dirs) {
            int cnt = 1;
            for (int mul = 1; mul < 3; mul++) {
                int *temp_coordinate = new int[3];
                temp_coordinate[0] = l + dir[0] * mul;
                temp_coordinate[1] = i + dir[1] * mul;
                temp_coordinate[2] = j + dir[2] * mul;
                if (!boundary_test(temp_coordinate))
                    break;
                if (temp_state[temp_coordinate[0]][temp_coordinate[1]][temp_coordinate[2]] != c)
                    break;
                cnt += 1;
            }
            for (int mul = 1; mul < 3; mul++) {
                int *temp_coordinate = new int[3];
                temp_coordinate[0] = l + dir[0] * -mul;
                temp_coordinate[1] = i + dir[1] * -mul;
                temp_coordinate[2] = j + dir[2] * -mul;
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
                } else {
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
    int board[6][6][6]{};
    int hands;
    Movement move;
    int visiting_count;
    int value_sum;
    bool is_leaf;
    bool is_root;
    vector<Node *> children;
    Properties my_properties;
    Node *parent{};


    Node(int board[6][6][6],
         int hands,
         Movement move,
         Properties my_properties) {

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


    Node *select() {
        Node *cur = this;
        while (true) {
            if (cur->is_leaf)
                return cur;
            else {
                double p = -1;
                Node *next_node = cur;
                for (int i = 0; i < cur->children.size(); i++) {
                    Node *temp_node = cur->children[i];
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
        vector<Movement> legal_moves = this->get_next_possible_move();
        int color;
        if (this->hands % 2 == 0)
            color = 1;
        else
            color = 2;
        for (auto &legal_move : legal_moves) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    for (int k = 0; k < 6; k++) {
                        temp_board[i][j][k] = this->board[i][j][k];
                    }
                }
            }
            Movement temp_move;
            temp_move = legal_move;
            temp_move.color = color;
            vector<Movement> movements;
            movements.clear();
            movements.push_back(temp_move);
            Properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
            temp_board[temp_move.l][temp_move.x][temp_move.y] = color;
            Node *new_child;
            new_child = new Node(temp_board, this->hands + 1, temp_move, new_properties);
            this->children.push_back(new_child);
            new_child->parent = this;

        }
    }


    int playout() {
        int temp_board[6][6][6];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    temp_board[i][j][k] = this->board[i][j][k];
                }
            }
        }
        vector<Movement> movements;
        movements.clear();
        for (int i = hands; i < 64; i++) {
            vector<Movement> legal_moves = this->get_next_possible_move();
            int random_number = rand() % legal_moves.size();
            Movement temp_move = legal_moves[random_number];
            if (i % 2 == 0) {
                temp_move.color = 1;
                temp_board[temp_move.l][temp_move.x][temp_move.y] = 1;
            } else {
                temp_move.color = 2;
                temp_board[temp_move.l][temp_move.x][temp_move.y] = 2;
            }
            movements.push_back(temp_move);
        }
        Properties end_properties = get_state_properties_b(board, my_properties, movements);

        if (hands % 2 == 0) {
            if (end_properties.black_points > end_properties.white_points)
                return 1;
            else
                return 0;
        } else {
            if (end_properties.white_points > end_properties.black_points)
                return 1;
            else
                return 0;
        }
    }


    void backup(int reward) {
        Node *cur = this;
        bool flag = true;
        if (reward == 1) {
            while (true) {
                if (flag) {
                    cur->value_sum += reward;
                    flag = false;
                } else {
                    flag = true;
                }
                cur->visiting_count++;
                if (cur->is_root)
                    break;
                cur = cur->parent;
            }
        } else {
            while (true) {
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


    vector<Movement> get_next_possible_move() {
        vector<Movement> ret;
        ret.clear();
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                if (this->board[5][i][j] != 0)
                    continue;
                for (int l = 0; l < 6; l++) {
                    if (this->board[l][i][j] == 0) {
                        Movement move(l, i, j, 0);
                        ret.push_back(move);
                        break;
                    }
                }
            }
        }
        return ret;
    }


    vector<Movement> gen_block_move() {
        vector<Movement> all_possible_moves = this->get_next_possible_move();
        int opponent_color = (this->hands % 2 == 0) ? 2 : 1;
        int dirs[26][3];
        int cnt = 0;
        for (int l = -1; l <= 1; l++) {
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (l == 0 && i == 0 && j == 0)
                        continue;
                    dirs[cnt][0] = l;
                    dirs[cnt][1] = i;
                    dirs[cnt][2] = j;
                    cnt++;
                }
            }
        }
        vector<Movement> block_moves;
        for (int move_num = 0; move_num < all_possible_moves.size(); move_num++) {
            for (int dir_num = 0; dir_num < 26; dir_num++) {
                int *new_pos = new int[3];
                new_pos[0] = all_possible_moves[move_num].l + dirs[dir_num][0];
                new_pos[1] = all_possible_moves[move_num].x + dirs[dir_num][1];
                new_pos[2] = all_possible_moves[move_num].y + dirs[dir_num][2];

                int l = new_pos[0], i = new_pos[1], j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] != opponent_color)
                    continue;
                new_pos[0] += dirs[dir_num][0];
                new_pos[1] += dirs[dir_num][1];
                new_pos[2] += dirs[dir_num][2];
                l = new_pos[0];
                i = new_pos[1];
                j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] != opponent_color)
                    continue;

                Movement block_move;
                block_move.l = all_possible_moves[move_num].l;
                block_move.x = all_possible_moves[move_num].x;
                block_move.y = all_possible_moves[move_num].y;
                block_move.color = (this->hands % 2 == 0) ? 1 : 2;
                block_moves.push_back(block_move);
            }
        }
        srand(time(0));
        if (!block_moves.empty()) {
            return block_moves;
        } else {
            return all_possible_moves;
        }
    }

    Node *get_node_after_playing(Movement next_move) {
        int temp_board[6][6][6];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    temp_board[i][j][k] = this->board[i][j][k];
                }
            }
        }
        int color = (hands % 2 == 0) ? 1 : 2;
        next_move.color = color;
        vector<Movement> movements;
        movements.clear();
        movements.push_back(next_move);
        Properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
        temp_board[next_move.l][next_move.x][next_move.y] = color;
        Node *ret = new Node(temp_board, this->hands + 1, next_move, new_properties);
        return ret;
    }

    void output_board_string_for_plot_state() {
        string board_str;
        for (auto &d : board) {
            for (auto &i : d) {
                for (int j : i) {
                    board_str += " " + to_string(j);
                }
            }
        }
        board_str += "\n";
        string output_filename = "board_content_for_plotting.txt";
        ofstream ofs(output_filename, fstream::trunc);
        ofs << board_str;
        ofs.close();
        cout << "Output board string to \"" << output_filename << "\" done!" << endl;
    }

    void print_flattened_board() {
        for (auto &l : this->board) {
            for (auto &x : l) {
                for (int y : x) {
                    cout << y << " ";
                }
            }
        }
        cout << endl;
    }
};

class MCTS {
public:
    Node *root;
    int cur_simulation_cnt;
    int max_simulation_cnt;
    int max_time_sec;
    bool print_simulation_cnt;

    MCTS(Node *root, int max_simulation_cnt, int max_time_sec, bool print_simulation_cnt) {
        this->root = root;
        this->max_simulation_cnt = max_simulation_cnt;
        this->max_time_sec = max_time_sec;
        this->cur_simulation_cnt = 0;
        this->print_simulation_cnt = print_simulation_cnt;
    }

    Movement run() {
        this->cur_simulation_cnt = 0;
        this->root->is_root = true;

        // Time example:
        // auto t0 = Time::now();
        // auto t1 = Time::now();
        // sec duration = t1 - t0;
        // std::cout << duration.count() << "s\n";

        // clock_t start = clock();
        auto start = Time::now();
        while (true) {
            //cout << 1 << endl;
            Node *temp_node = this->root->select();
            //cout << 2 << endl;
            temp_node->expand();
            //cout << 3 << endl;
            int reward = temp_node->playout();
            //cout << 4 << endl;
            temp_node->backup(reward);
            this->cur_simulation_cnt++;
            // clock_t end = clock();
            auto end = Time::now();
            sec duration = Time::now() - start;
            if (duration.count() >= (double) (this->max_time_sec))
                break;
            if (this->cur_simulation_cnt >= this->max_simulation_cnt)
                break;
        }
        double winning_rate = 0;
        Movement ret;
        for (auto temp_node : this->root->children) {
            if (temp_node->visiting_count == 0)
                continue;
            //cout << temp_node->value_sum << " " << temp_node->visiting_count << endl;
            double temp_winning_rate = (double) temp_node->value_sum / (double) temp_node->visiting_count;
            //cout << temp_winning_rate << endl;
            if (temp_winning_rate > winning_rate) {
                winning_rate = temp_winning_rate;
                ret = temp_node->move;
                //cout << temp_node->move.l << " " << temp_node->move.x << " " << temp_node->move.y << endl;
            }
        }
        if (this->print_simulation_cnt) {
            cout << "Simulation cnt: " << this->cur_simulation_cnt << endl;
        }
        return ret;
    }

    static Node *get_init_node() {
        int b[6][6][6];
        for (auto &i : b) {
            for (auto &j : i) {
                for (int &k : j) {
                    k = 0;
                }
            }
        }
        for (auto &k : b) {
            k[0][0] = -1;
            k[0][1] = -1;
            k[0][4] = -1;
            k[0][5] = -1;
            k[1][0] = -1;
            k[1][5] = -1;
            k[4][0] = -1;
            k[4][5] = -1;
            k[5][0] = -1;
            k[5][1] = -1;
            k[5][4] = -1;
            k[5][5] = -1;
        }

        Properties start_properties(0, 0, 0, 0);
        Movement move, next_move;
        Node *start_node = new Node(b, 0, move, start_properties);
        return start_node;
    }

    static Node *get_random_board_node(int step) {
        Node *cur_node = MCTS::get_init_node();
        cout << "Getting random board..." << endl;
        cout << "Move done:";
        for (int i = 0; i < step; i++) {
            MCTS mcts(cur_node, 99999, 1, false);
            Movement move = mcts.run();
            cout << " " << i + 1 << flush;
            Node *new_node = cur_node->get_node_after_playing(move);
            delete cur_node;
            cur_node = new_node;
        }
        cout << " (finished)" << endl;
        return cur_node;
    }
};

int main() {
    // Init random
    srand(time(nullptr));

//    Node *cur_node = MCTS::get_init_node();
    Node *cur_node = MCTS::get_random_board_node(10);
    cur_node->output_board_string_for_plot_state();


//
//    vector<Movement> v = start_node->gen_block_move();
//    for (auto &move: v) {
//        cout << move.l << ", " << move.x << ", " << move.y << endl;
//    }
//    cout << "Total: " << v.size() << " moves." << endl;

//    MCTS mcts(start_node, 9999999, 3);
//    Movement next_move = mcts.run();
//    cout << next_move.l << " " << next_move.x << " " << next_move.y << endl;
}
