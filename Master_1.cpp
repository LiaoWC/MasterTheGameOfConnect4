#include "STcpClient.h"
#include<iostream>
#include <utility>
#include<vector>
#include<cmath>
#include<ctime>
#include <chrono>
#include <cstdlib>
// #include <unistd.h> // Comment when on Windows
#include <fstream>
#include <memory>
#include <array>
#include <algorithm>
#include <iomanip>

#define NEG_INFINITE -9999999
#define POS_INFINITE 9999999
#define ILLEGAL_COLOR = -1
#define EMPTY_COLOR 0
#define BLACK_PLAYER_COLOR 1
#define WHITE_PLAYER_COLOR 2
// For expand's return value
#define EXPAND_NOT_PRUNE_PARENT 1
#define EXPAND_PRUNE_PARENT 2
#define EXPAND_ROOT_HAS_DOMINANT_MOVE 3
//

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> sec;
using namespace std;


class Movement;

bool operator==(const Movement &m1, const Movement &m2);

class Movement {
public:
    int l{};
    int x{};
    int y{};
    int color{};
    double prior{};
    double base_c{};

    Movement() {
        this->l = 0;
        this->x = 0;
        this->y = 0;
        this->color = 0;
        this->prior = 0;
        this->base_c = 0.05;
    };

    Movement(int l, int x, int y, int color) {
        this->l = l;
        this->x = x;
        this->y = y;
        this->color = color;
        this->prior = 0;
        this->base_c = 0.05;
    }

    void config_c(double newc) {
        this->base_c = newc;
    }

    void print_movement() const {
        cout << "[" << l << ", " << x << ", " << y << "]" << endl;
    }

    friend bool operator==(const Movement &m1, const Movement &m2);
};


bool operator==(const Movement &m1, const Movement &m2) {
    // TODO: get_all_possible_move function seems to give no player color information. It gives zero always?
    // To prevent some unexpected color occurring, we only check l, i, j
    if (m1.l == m2.l && m1.x == m2.x && m1.y == m2.y) { return true; } else { return false; }
}


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

    void print_properties() const {
        cout << "Bp: " << black_points << ", Wp: " << white_points << ", Bl: " << black_lines << ", Wl: " << white_lines
             << endl;
    }

    void output_properties() const {
        ofstream ofs("output_properties_for_plotting.txt", fstream::trunc);
        ofs << "BlackPoints: " << black_points << ", WhitePoints: " << white_points << ", BlackLines: " << black_lines
            << ", WhiteLines: " << white_lines;
        ofs.close();
        cout << "Output properties string to \"output_properties_for_plotting.txt\" done!";
    }
};

bool boundary_test(const unique_ptr<int[]> &coordinate) {
    for (int i = 0; i < 3; i++)
        if (coordinate[i] < 0 || coordinate[i] >= 6)
            return false;
    return true;
}

int pre_state[6][6][6] = {};
Properties pre_state_properties(0, 0, 0, 0);

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
            for (int mul = 1; mul <= 3; mul++) {
                unique_ptr<int[]> temp_coordinate(new int[3]);
                temp_coordinate[0] = l + dir[0] * mul;
                temp_coordinate[1] = i + dir[1] * mul;
                temp_coordinate[2] = j + dir[2] * mul;
                if (!boundary_test(temp_coordinate))
                    break;
                if (temp_state[temp_coordinate[0]][temp_coordinate[1]][temp_coordinate[2]] != c)
                    break;
                cnt += 1;
            }
            for (int mul = 1; mul <= 3; mul++) {
                unique_ptr<int[]> temp_coordinate(new int[3]);
                temp_coordinate[0] = l + dir[0] * -mul;
                temp_coordinate[1] = i + dir[1] * -mul;
                temp_coordinate[2] = j + dir[2] * -mul;
                if (!boundary_test(temp_coordinate))
                    break;
                if (temp_state[temp_coordinate[0]][temp_coordinate[1]][temp_coordinate[2]] != c)
                    break;
                cnt += 1;
            }
            //cout << cnt << endl;
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

Properties get_state_properties_a(int now_state[6][6][6], bool is_black){
    int temp_state[6][6][6];
    for(int l = 0; l < 6 ; l++){
        for(int i = 0; i < 6 ; i++){
            for(int j = 0; j < 6; j++){
                temp_state[l][i][j] = now_state[l][i][j] - pre_state[l][i][j];             
            }
        }
    }
	vector<Movement> movements(2);

    for(int l = 0; l < 6 ; l++){
        for(int i = 0; i < 6 ; i++){
            for(int j = 0; j < 6; j++){
                if (temp_state[l][i][j] == 1){
                    int r = (is_black)?1:0;
					movements[r] = Movement{l, i, j, 1};
                }
                else if (temp_state[l][i][j] == 2){
                    int r = (is_black)?0:1;
					movements[r] = Movement{l, i, j, 2};
                }
            }
        }
    }
    Properties ret = get_state_properties_b(pre_state, pre_state_properties, movements);
    return ret;
}
class Node : public std::enable_shared_from_this<Node> {
public:
    int board[6][6][6]{};
    int hands;
    Movement move;
    int visiting_count;
    int value_sum;
    bool is_leaf;
    bool is_root;
    bool is_terminated; // Prevent from doing expanding
    vector<shared_ptr<Node>> children;
    Properties my_properties;
    weak_ptr<Node> parent;

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
        this->is_terminated = false;
        children.clear();
    }

    ~Node() {
        children.clear();
    }

    static double ucb(int child_visiting_count,
                      int winning_count,
                      double parent_visit_cnt) {
        if (child_visiting_count == 0)
            child_visiting_count = 1;
        if (parent_visit_cnt == 0)
            parent_visit_cnt = 1;
        double q, u;
        u = log10((double) parent_visit_cnt) / (double) child_visiting_count;
        u = pow(u, 0.5) * 1.414;
        q = (double) winning_count / (double) child_visiting_count;
        return q + u;
    }

    // PUCT: Predictor + UCB applied to trees
    // TODO: Make it easy to tune the hyper-parameters
    static double puct(int child_visiting_count,
                       int winning_count,
                       double parent_visit_cnt, double prior_value) {
        if (child_visiting_count == 0)
            child_visiting_count = 1;
        if (parent_visit_cnt == 0)
            parent_visit_cnt = 1;
        double q, u;
        // Exploration
        q = (double) winning_count / (double) child_visiting_count;
        // Exploitation
        double puct_constant = 1.414;
        u = puct_constant * (pow((double) parent_visit_cnt, 0.5) / (double) (1 + child_visiting_count));
        u += prior_value;
        // PUCT = Q(s,a) + U(s,a)
        return q + u;
    }

    shared_ptr<Node> select(bool prior_value) {
        shared_ptr<Node> cur = shared_from_this();
        while (true) {
            if (cur->is_leaf)
                return cur;
            else {
                // double cur_max_ucb_result = -1;
                double cur_max_ucb_result = NEG_INFINITE;
                vector<double> ucb_results;
                for (unsigned int i = 0; i < cur->children.size(); i++) {
                    shared_ptr<Node> temp_node = cur->children[i];
                    double ucb_result = NEG_INFINITE;
                    if (cur->is_root && temp_node->visiting_count < 1) {
                        ucb_result = POS_INFINITE;
                    } else {
                        if (prior_value) {
                            //  (-1) * value sum is the most important
                            ucb_result =
                                    puct(temp_node->visiting_count, (-1) * temp_node->value_sum, cur->visiting_count,
                                         temp_node->move.prior);
                        } else {
                            ucb_result =
                                    ucb(temp_node->visiting_count, (-1) * temp_node->value_sum, cur->visiting_count);
                        }
                    }

                    ucb_results.emplace_back(ucb_result);
                    if (ucb_result > cur_max_ucb_result) {
                        cur_max_ucb_result = ucb_result;
                    }
                }
                vector<unsigned int> max_value_idx;
                for (unsigned int i = 0; i < cur->children.size(); i++) {
                    if (ucb_results[i] == cur_max_ucb_result) {
                        max_value_idx.emplace_back(i);
                    }
                }
                if (max_value_idx.empty()) {
                    // We don't know if some nodes that has no children will be selected due to any reasons,
                    // so to make it safer, if the node selected isn't a leaf node but still has no child,
                    // we just return it as the select result.
                    return cur;
                }
                int random_number = rand() % max_value_idx.size();
                shared_ptr<Node> next_node = cur->children[max_value_idx[random_number]];
                cur = next_node;
            }
        }
    }

    int expand(bool dominant_pruning, bool prior_value, bool not_check_dominant_on_first_layer) {
        this->is_leaf = false;
        int temp_board[6][6][6];
        vector<Movement> legal_moves;
        if (prior_value) {
            legal_moves = this->gen_block_move();
        } else {
            legal_moves = this->get_next_possible_move();
        }
        vector<int> dominant_move_indices;
        shared_ptr<Node> parent_shared_ptr;
        // Check if parent is root
        bool parent_node_is_root = false;
        if (!this->is_root) {
            parent_shared_ptr = parent.lock();
            parent_node_is_root = parent_shared_ptr->is_root;
        }

        for (int legal_move_idx = 0; legal_move_idx < legal_moves.size(); legal_move_idx++) {
            Movement legal_move = legal_moves[legal_move_idx];
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    for (int k = 0; k < 6; k++) {
                        temp_board[i][j][k] = this->board[i][j][k];
                    }
                }
            }
            Movement temp_move;
            temp_move = legal_move;
            int color = this->hands % 2 == 0 ? 1 : 2;

            temp_move.color = color;
            vector<Movement> movements;
            movements.clear();
            movements.push_back(temp_move);


            Properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
            temp_board[temp_move.l][temp_move.x][temp_move.y] = color;

            if (dominant_pruning && !(parent_node_is_root && not_check_dominant_on_first_layer)) {
                // Check if it is a dominant move or a regular move.
                bool is_dominant = false;
                if ((color == BLACK_PLAYER_COLOR && (new_properties.black_lines - my_properties.black_lines > 0))
                    || (color = WHITE_PLAYER_COLOR && (new_properties.white_lines - my_properties.white_lines > 0))) {
                    is_dominant = true;
                }

                if (is_dominant) {
                    if (!is_root) {
                        // Check if parent will have no children after this child being pruned
                        if (parent_shared_ptr->children.size() == 1) {
                            // If so, make parent node be terminated and be a leaf node
                            parent_shared_ptr->is_terminated = true;
                            parent_shared_ptr->is_leaf = true;
                        }
                        // Remove this child from parent
                        int idx = 0;
                        for (; idx < parent_shared_ptr->children.size(); idx++) { // Find where this child is
                            if (parent_shared_ptr->children[idx]->move.x == this->move.x &&
                                parent_shared_ptr->children[idx]->move.y == this->move.y &&
                                parent_shared_ptr->children[idx]->move.l == this->move.l) {
                                // Find it!
                                break;
                            }
                        }
                        parent_shared_ptr->children.erase(parent_shared_ptr->children.begin() + idx);
//                        cout << "Hola" << endl;
                        if (parent_node_is_root) {
                            cout << "&&&&&&&&&&&&&&&&& prunedddd    " << endl;
                        }
                        return EXPAND_PRUNE_PARENT;
                    } else { // Root node
                        dominant_move_indices.emplace_back(legal_move_idx);
                    }
                }
            }
            shared_ptr<Node> new_child = make_shared<Node>(temp_board, this->hands + 1, temp_move, new_properties);
            this->children.push_back(new_child);
            new_child->parent = shared_from_this();
        }

        // For root node: If there are any dominant moves existing, we prune all non-dominant nodes.
        if (this->is_root && dominant_pruning && (!dominant_move_indices.empty())) {
            vector<shared_ptr<Node>> children_pruned;
            // "Better performance will be reached when avoiding dynamic reallocation, so try to have the vector memory be big enough to receive all elements."
            //  -- reference: https://stackoverflow.com/questions/32199388/what-is-better-reserve-vector-capacity-preallocate-to-size-or-push-back-in-loo
            children_pruned.reserve(dominant_move_indices.size());
            for (int idx: dominant_move_indices) {
                children_pruned.emplace_back(this->children[idx]);
            }
            this->children = children_pruned;
            return EXPAND_ROOT_HAS_DOMINANT_MOVE;
        }
        return EXPAND_NOT_PRUNE_PARENT;
    }

    int playout(bool use_block_moves) {
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
        for (int i = this->hands; i < 64; i++) {
            vector<Movement> considered_moves;
            // Use block move to filter moves we consider
            // The block_moves function will return all legal moves if find no block moves
            if (use_block_moves) {
                considered_moves = gen_block_move();
            } else {
                considered_moves = this->get_next_possible_move();
            }
            // TODO: Check and modify and let block move gotten are all legal moves.
            int random_number = rand() % considered_moves.size();
            Movement temp_move = considered_moves[random_number];
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
        if (this->is_root) {
            cout << "Playout hands: " << this->hands << endl;
        }
        if (auto ppp = this->parent.lock()) {
            if (ppp->is_root) {
                //cout << "Playout hands: " << this->hands << endl;
            }
        }
        if (this->hands % 2 == 0) {
            if (end_properties.black_points > end_properties.white_points)
                return 1;
            else if (end_properties.black_points < end_properties.white_points)
                return -1;
            else
                return 0;
        } else {
            if (end_properties.white_points > end_properties.black_points)
                return 1;
            else if ((end_properties.white_points < end_properties.black_points))
                return -1;
            else
                return 0;
        }
    }

    void backup(int reward, bool use_reverse_for_opponent) {
        shared_ptr<Node> cur = shared_from_this();
        bool flag = true;
        if (!use_reverse_for_opponent) {
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
                    // Weak ptr
                    if (auto p = cur->parent.lock()) {
                        cur = p;
                    } else {
                        perror("Backup function cur->parent.lock() No.1 failed.");
                    }
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
                    // Weak ptr
                    if (auto p = cur->parent.lock()) {
                        cur = p;
                    } else {
                        perror("Backup function cur->parent.lock() No.2 failed.");
                    }
                }
            }
        } else {
            while (true) {
//                if(this->move.l==0&& this->move.x==2&&this->move.y==0 && this->children.){
//                    cout << (flag?"me":"opponent")<<endl;
//                }

                if (flag) {
                    cur->value_sum += reward;
                    flag = false;
                } else {
                    cur->value_sum -= reward;
                    flag = true;
                }
                cur->visiting_count++;
                if (cur->is_root)
                    break;
                // Weak ptr
                if (auto p = cur->parent.lock()) {
                    cur = p;
                } else {
                    perror("Backup function cur->parent.lock() No.3 failed.");
                }
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
                        Movement move_tmp(l, i, j, 0);
                        ret.push_back(move_tmp);
                        break;
                    }
                }
            }
        }
        return ret;
    }

    vector<Movement> gen_block_move() {
        int temp_state[6][6][6];
        for (int l = 0; l < 6; l++) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    temp_state[l][i][j] = this->board[l][i][j];
                }
            }
        }
        bool flag = true;
        bool flag2 = true;
        vector<Movement> all_possible_moves = this->get_next_possible_move();
        int opponent_color = (this->hands % 2 == 0) ? 2 : 1;
        int self_color = (this->hands % 2 == 0) ? 1 : 2;
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
        int dirs2[13][3] = {
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
        vector<Movement> block_moves;
        for (auto &all_possible_move : all_possible_moves) {
            double temp_p = 0;
            flag = true;
            flag2 = true;
            for (auto &dir : dirs) {
                unique_ptr<int[]> new_pos(new int[3]);
                new_pos[0] = all_possible_move.l + dir[0];
                new_pos[1] = all_possible_move.x + dir[1];
                new_pos[2] = all_possible_move.y + dir[2];

                int l = new_pos[0];
                int i = new_pos[1];
                int j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] != opponent_color)
                    continue;
                new_pos[0] += dir[0];
                new_pos[1] += dir[1];
                new_pos[2] += dir[2];
                l = new_pos[0];
                i = new_pos[1];
                j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] != opponent_color)
                    continue;
                new_pos[0] += dir[0];
                new_pos[1] += dir[1];
                new_pos[2] += dir[2];
                l = new_pos[0];
                i = new_pos[1];
                j = new_pos[2];
                if (!boundary_test(new_pos) || this->board[l][i][j] == self_color)
                    continue;
                Movement block_move;
                block_move.l = all_possible_move.l;
                block_move.x = all_possible_move.x;
                block_move.y = all_possible_move.y;
                block_move.color = self_color;
                block_move.prior += block_move.base_c;
                block_moves.push_back(block_move);
                flag = false;
            }
            int l = all_possible_move.l;
            int i = all_possible_move.x;
            int j = all_possible_move.y;
            //cout << l << " " << i << " " << j << " " << c << endl;
            for (auto &dir2 : dirs2) {
                int cnt = 1;
                for (int mul = 1; mul <= 2; mul++) {
                    unique_ptr<int[]> new_pos(new int[3]);
                    new_pos[0] = l + dir2[0] * mul;
                    new_pos[1] = i + dir2[1] * mul;
                    new_pos[2] = j + dir2[2] * mul;
                    if (!boundary_test(new_pos))
                        break;
                    if (temp_state[new_pos[0]][new_pos[1]][new_pos[2]] != self_color)
                        break;
                    cnt += 1;
                }
                for (int mul = 1; mul <= 2; mul++) {
                    unique_ptr<int[]> new_pos(new int[3]);
                    new_pos[0] = l + dir2[0] * -mul;
                    new_pos[1] = i + dir2[1] * -mul;
                    new_pos[2] = j + dir2[2] * -mul;
                    if (!boundary_test(new_pos))
                        break;
                    if (temp_state[new_pos[0]][new_pos[1]][new_pos[2]] != self_color)
                        break;
                    cnt += 1;
                }
                //cout << "cnt: " << cnt << endl;
                while (cnt >= 3) {
                    Movement temp_block_move;
                    temp_p += temp_block_move.base_c * 2;
                    flag2 = false;
                    cnt--;
                }
            }
            if (!flag2) {
                Movement block_move;
                block_move.l = all_possible_move.l;
                block_move.x = all_possible_move.x;
                block_move.y = all_possible_move.y;
                block_move.color = self_color;
                block_move.prior = temp_p;
                block_moves.push_back(block_move);
            }
            if (flag && flag2) {
                Movement block_move;
                block_move.l = all_possible_move.l;
                block_move.x = all_possible_move.x;
                block_move.y = all_possible_move.y;
                block_move.color = self_color;
                block_moves.push_back(block_move);
            }
        }
        srand(time(nullptr));
        return block_moves;
    }

    shared_ptr<Node> get_node_after_playing(Movement next_move) {
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
        cout << "aaaaaaaaaaaaaa  " << this->hands << endl;
        shared_ptr<Node> ret = make_shared<Node>(temp_board, this->hands + 1, next_move, new_properties);
        return ret;
    }

    // void output_board_string_for_plot_state() {
    //     string board_str;
    //     for (auto &d : board) {
    //         for (auto &i : d) {
    //             for (int j : i) {
    //                 board_str += " " + to_string(j);
    //             }
    //         }
    //     }
    //     board_str += "\n";
    //     string output_filename = "board_content_for_plotting.txt";
    //     ofstream ofs(output_filename, fstream::trunc);
    //     ofs << board_str;
    //     ofs.close();
    //     cout << "Output board string to \"" << output_filename << "\" done!" << endl;
    // }

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

private:
    // vector<vector<float>> first_layer_stat(int mode, const string &output_filename_suffix) {
    //     // If output_filename_suffix == empty string means:
    //     //     "not to output to file"
    //     // Mode
    //     //     1: visiting_count
    //     //     2: value_sum
    //     //     3: visiting_count / value_sum
    //     vector<vector<float>> plane(6);
    //     string mode_name;
    //     for (auto &row: plane) {
    //         row.resize(6);
    //     }
    //     for (auto &child: root->children) {
    //         float value;
    //         switch (mode) {
    //             case 1:
    //                 value = (float) child->visiting_count;
    //                 mode_name = "Visiting_Count";
    //                 break;
    //             case 2:
    //                 value = (float) child->value_sum;
    //                 mode_name = "Value_Sum";
    //                 break;
    //             case 3:
    //                 value = (float) child->value_sum / (float) child->visiting_count;
    //                 mode_name = "Value_Mean";
    //                 break;
    //             default:
    //                 cerr << "Invalid mode: " << mode << endl;
    //                 exit(EXIT_FAILURE);
    //         }
    //         plane[child->move.x][child->move.y] = value;
    //     }
    //     // Output file
    //     if (!output_filename_suffix.empty()) {
    //         ofstream ofs("first_layer_stat_" + output_filename_suffix + ".txt", fstream::trunc);
    //         ofs << mode_name;
    //         for (const auto &line: plane) {
    //             for (auto item: line) {
    //                 ofs << " " << item;
    //             }
    //         }
    //         ofs.close();
    //     }
    //     // Return 2D vector
    //     return plane;
    // }

public:
    shared_ptr<Node> root;
    int cur_simulation_cnt;
    int max_simulation_cnt;
    double max_time_sec;
    bool print_simulation_cnt;

    MCTS(shared_ptr<Node> root, int max_simulation_cnt, double max_time_sec, bool print_simulation_cnt) {
        this->root = std::move(root);
        this->max_simulation_cnt = max_simulation_cnt;
        this->max_time_sec = max_time_sec;
        this->cur_simulation_cnt = 0;
        this->print_simulation_cnt = print_simulation_cnt;
    }

    static Movement get_rand_first_hand_center_move(bool random) {
        if (random) {
            int rand_num = rand() % 4;
            switch (rand_num) {
                case 0:
                    return {0, 2, 2, 1};
                case 1:
                    return {0, 2, 3, 1};
                case 2:
                    return {0, 3, 2, 1};
                case 3:
                    return {0, 3, 3, 1};
                default: // If error we direct
                    break;
            }
        }
        return {0, 2, 2, 1}; // Since the board is reflexive and mirror, we may have no need to random
    }


    Movement run(bool playout_use_block_move,
                 bool reward_add_score_diff,
                 bool first_hand_center,
                 bool dominate_pruning,
                 bool prior_value,
                 bool not_check_dominate_pruning_on_first_layer) {

        // If it is first hand (black)
        if (first_hand_center && root->hands == 0 && this->root->board[0][2][2] == 0) {
            return {0, 2, 2, 1};
        }


        this->cur_simulation_cnt = 0;
        this->root->is_root = true;

        bool root_has_dominant_move = false;

        // Time example:
        // auto t0 = Time::now();
        // auto t1 = Time::now();
        // sec duration = t1 - t0;
        // std::cout << duration.count() << "s\n";

        // clock_t start = clock();
        auto start = Time::now();
        while (true) {
            auto end = Time::now();
            sec duration = Time::now() - start;
            if (duration.count() >= (double) (this->max_time_sec))
                break;
            if (this->cur_simulation_cnt >= this->max_simulation_cnt)
                break;
            /////////////////////////////////////////////////////
            // SELECT
            /////////////////////////////////////////////////////
            shared_ptr<Node> temp_node = this->root->select(prior_value);
            // Check if root's children are all pruned
            if (temp_node->is_root && temp_node->is_terminated) {
                break;
            }
            // Check if selected node is terminated (e.g. hands == 64 or other factors)
            if (temp_node->hands >= 64) { // Number of hands reaches limit.
                temp_node->is_terminated = true;
            }
            /////////////////////////////////////////////////////
            // EXPAND
            /////////////////////////////////////////////////////
            if (!temp_node->is_terminated) {
                int expand_rt = temp_node->expand(dominate_pruning, prior_value,
                                                  not_check_dominate_pruning_on_first_layer);
                // expand_rt==EXPAND_NOT_PRUNE_PARENT is ok
                if (expand_rt == EXPAND_PRUNE_PARENT) {
                    continue; // Drop this simulation
                } else if (expand_rt == EXPAND_ROOT_HAS_DOMINANT_MOVE) {
                    root_has_dominant_move = true;
                }
            }
            ///////////////////////////////////////////////////
            // EVALUATE
            ///////////////////////////////////////////////////
            int reward = 0;
            if (temp_node->is_terminated) {
                int color = (temp_node->hands % 2 == 0) ? BLACK_PLAYER_COLOR : WHITE_PLAYER_COLOR;
                // 1: black win; 2: white win; 0: draw
                int black_win = 0;
                if (temp_node->my_properties.black_points > temp_node->my_properties.white_points) {
                    black_win = 1;
                } else if (temp_node->my_properties.black_points < temp_node->my_properties.white_points) {
                    black_win = 2;
                }
                if (black_win == 0) {
                    reward = 0;
                } else if (black_win == 1) {
                    if (color == BLACK_PLAYER_COLOR) {
                        reward = 1;
                    } else {
                        reward = -1;
                    }
                } else {
                    if (color == BLACK_PLAYER_COLOR) {
                        reward = -1;
                    } else {
                        reward = 1;
                    }
                }
            } else {
                reward = temp_node->playout(playout_use_block_move);
            }
            // Add score_diff
            if (reward_add_score_diff && !temp_node->is_root) {
                if (auto ptr = temp_node->parent.lock()) {
                    Properties parent_pro = ptr->my_properties;
                    if (temp_node->hands % 2 ==
                        0) { // Black // TODO: Here seems to be wrong. "hands" cannot be used in MCTS
                        reward += (temp_node->my_properties.black_points - parent_pro.black_points) -
                                  (temp_node->my_properties.white_points - parent_pro.white_points);
                    } else { // White
                        reward += (temp_node->my_properties.white_points - parent_pro.white_points) -
                                  (temp_node->my_properties.black_points - parent_pro.black_points);
                    }
                } else {
                    perror("MCTS run reward_add_score_diff get weak_ptr .lock() failed.");
                }
            }
            ///////////////////////////////////////////////////
            // BACKUP
            ///////////////////////////////////////////////////
            //            cout << 4 << endl;
            temp_node->backup(reward, true);
            this->cur_simulation_cnt++;
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        // Re run MCTS if root's children are all pruned but not because root has any dominant moves
        /////////////////////////////////////////////////////////////////////////////////////////////
        if (!root_has_dominant_move && this->root->is_terminated) {
            shared_ptr<Node> new_root = make_shared<Node>(this->root->board, this->root->hands, this->root->move,
                                                          this->root->my_properties);
            auto end = Time::now();
            sec duration = Time::now() - start;
            cout << "Has used " << duration.count() << " sec" << endl;
            double rest_of_time = ((double) (this->max_time_sec) - duration.count());
            double rerun_time = max(1., rest_of_time);
            MCTS mcts(new_root, 999999, rerun_time, true);
            cout << "Re-mcts-run with remained time: " << rerun_time << " sec." << endl;
            // Run mcts with simple settings
            cout << new_root->hands << " DDDDDDDDDDDDD" << endl;
            Movement new_move = mcts.run(false, false, false, false, false, false);
            this->root = new_root;
            return new_move;
        }

        if (this->print_simulation_cnt) {
            cout << "Simulation cnt: " << this->cur_simulation_cnt << endl;
        }

        /////////////////////////////////////////////////
        // PICK A MOVE
        /////////////////////////////////////////////////
        double winning_rate = NEG_INFINITE;
        Movement ret{-1, -1, -1, -1};
        for (const auto &temp_node : this->root->children) {
            if (temp_node->visiting_count == 0) {
                continue;
            }
            // TODO: it seems there's node that ought to be pruned but still be picked up for playing next move
            //cout << temp_node->value_sum << " " << temp_node->visiting_count << endl;
            double value_mean = (double) temp_node->value_sum / (double) temp_node->visiting_count;
            //cout << temp_winning_rate << endl;
            if (value_mean > winning_rate) {
                winning_rate = value_mean;
                ret = temp_node->move;
                //cout << temp_node->move.l << " " << temp_node->move.x << " " << temp_node->move.y << endl;
            }
        }
        return ret;
    }

    static shared_ptr<Node> get_init_node() {
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
        shared_ptr<Node> start_node = make_shared<Node>(b, 0, move, start_properties);
        return start_node;
    }

    static shared_ptr<Node> get_random_board_node(int step, int max_simulation_cnt, int max_simulation_time) {
        shared_ptr<Node> cur_node = MCTS::get_init_node();
        cout << "Getting random board..." << endl;
        cout << "Move done:" << endl;
        for (int i = 0; i < step; i++) {
            MCTS mcts(cur_node, max_simulation_cnt, max_simulation_time, true);
            cout << i + 1 << " " << flush;
            Movement move = mcts.run(false, false, false, false, false, false);
            shared_ptr<Node> new_node = cur_node->get_node_after_playing(move);
            cur_node = new_node;
        }
        cout << " (finished) " << endl;
        return cur_node;
    }

    ~MCTS() {
        root.reset();
    }


    // vector<vector<float>> first_layer_visit_cnt_distribution(const string &output_filename_suffix) {
    //     // Output filename == empty string means "not to output to file"
    //     return first_layer_stat(1, output_filename_suffix);
    // }

    // vector<vector<float>> first_layer_value_sum_distribution(const string &output_filename_suffix) {
    //     // Output filename == empty string means "not to output to file"

    //     return first_layer_stat(2, output_filename_suffix);
    // }

    // vector<vector<float>> first_layer_value_mean_distribution(const string &output_filename_suffix) {
    //     // Output filename == empty string means "not to output to file"
    //     return first_layer_stat(3, output_filename_suffix);
    // }

};

/*
	輪到此程式移動棋子
	board : 棋盤狀態(vector of vector), board[l][i][j] = l layer, i row, j column 棋盤狀態(l, i, j 從 0 開始)
			0 = 空、1 = 黑、2 = 白、-1 = 四個角落
	is_black : True 表示本程式是黑子、False 表示為白子

	return Step
	Step : vector, Step = {r, c}
			r, c 表示要下棋子的座標位置 (row, column) (zero-base)
*/
std::vector<int> GetStep(std::vector<std::vector<std::vector<int>>> &board, bool is_black)
{
	srand(time(nullptr));

    int max_simulation_cnt = 9999999;
    int max_simulation_time = 1;
	int now_board[6][6][6];
	for(int i = 0; i < 6; i++){
		for(int j = 0 ; j < 6 ; j++){
			for(int k = 0 ; k < 6 ; k++){
				now_board[i][j][k] = board[i][j][k];
			}
		}
	}
	Movement move;
	Properties prop =  get_state_properties_a(now_board, is_black);
	shared_ptr<Node> cur_node = make_shared<Node>(now_board, 0, move, prop);
	if(is_black){
        // Black's turn
        MCTS mcts_black(cur_node, max_simulation_cnt, max_simulation_time, true);
        move = mcts_black.run(false, false, true, true, true, false);
		cur_node = cur_node->get_node_after_playing(move);
	}
	else{
        // White's turn
        MCTS mcts_white(cur_node, max_simulation_cnt, max_simulation_time, true);
        move = mcts_white.run(false, false, true, true, true, false);
		cur_node = cur_node->get_node_after_playing(move);
	}

	// Update pre_state
	
	for(int i = 0; i < 6; i++){
		for(int j = 0 ; j < 6 ; j++){
			for(int k = 0 ; k < 6 ; k++){
				pre_state[i][j][k] = board[i][j][k];
			}
		}
	}
	pre_state_properties = prop;
}

int main()
{
	int id_package;
	std::vector<std::vector<std::vector<int>>> board;
	std::vector<int> step;

	bool is_black;
	while (true)
	{
		if (GetBoard(id_package, board, is_black))
			break;

		step = GetStep(board, is_black);
		SendStep(id_package, step);
	}
}
