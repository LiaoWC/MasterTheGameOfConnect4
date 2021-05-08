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

#define  NEG_INFINITE -9999999
#define ILLEGAL_COLOR = -1
#define EMPTY_COLOR 0
#define BLACK_PLAYER_COLOR 1
#define WHITE_PLAYER_COLOR 2
// For expand's return value
#define EXPAND_NOT_PRUNE_PARENT 1
#define EXPAND_PRUNE_PARENT 2
#define EXPAND_ROOT_HAS_NO_CHILD_AFTER_PRUNED 3

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> sec;
using namespace std;

//////////////////////////////////////
class Movement;

bool operator==(const Movement &m1, const Movement &m2);
//////////////////////////////////////

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

template<typename T>
void print_vector_2d_plane(vector<vector<T>> plane) {
    cout << endl;
    for (const auto &line: plane) {
        for (const auto item: line) {
            cout << setw(14) << item;
        }
        cout << endl;
    }
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

    void output_properties() {
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


//Properties get_state_properties_b(array<array<array<int,6>,6>,6> start_state,
Properties get_state_properties_b(int start_state[6][6][6],
                                  Properties start_state_properties,
                                  const vector<Movement> &movements) {
    cout << "!!!!! SSSSSSSSSSSSTSRT " << endl;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                cout << setw(5) << start_state[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    start_state_properties.print_properties();
    cout << "&&&&&&&&&&&&&&&&&&&" << endl;
    for (auto item: movements) {
        item.print_movement();
    }
    cout << "&&&&&&&&&&&&&&&&&&&" << endl;

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
    cout << "@@@@@@@@@@@@@@@@" << endl;
    temp_properties.print_properties();
    cout << "@@@@@@@@@@@@@@" << endl;
    return temp_properties;
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
    bool is_pruned; // Prevent from being selected
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
        this->is_pruned = false;
        children.clear();
    }

    ~Node() {
        children.clear();
    }

    shared_ptr<Node> select() {
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
                    double ucb_result;
                    if (temp_node->is_pruned) {
                        ucb_result = NEG_INFINITE;
                    } else {
                        ucb_result = ucb(temp_node->visiting_count, temp_node->value_sum, cur->visiting_count);
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
                int random_number = rand() % max_value_idx.size();
                shared_ptr<Node> next_node = cur->children[max_value_idx[random_number]];
                cur = next_node;
            }
        }
    }

    int expand(bool dominant_pruning) {
        cout << endl << endl;
        cout << "ccccccccccccccccccccccccccccccccc" << endl;
        cout << "ccccccccccccccccccccccccccccccccc" << endl;
        cout << "ccccccccccccccccccccccccccccccccc" << endl;
        cout << "ccccccccccccccccccccccccccccccccc" << endl;
        cout << "ccccccccccccccccccccccccccccccccc" << endl;
        cout << "ccccccccccccccccccccccccccccccccc" << endl;
        cout << "ccccccccccccccccccccccccccccccccc" << endl;
        this->is_leaf = false;
        int temp_board[6][6][6];
        vector<Movement> legal_moves = this->get_next_possible_move();
        int color;
        cout << this->hands << endl;
        if (this->hands % 2 == 0) {
            cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaa" << endl;
            cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaa" << endl;
            cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaa" << endl;
            cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaa" << endl;
            cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaa" << endl;
            cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaa" << endl;
            color = 1;
        } else {
            cout << "bbbbbbbbbbbbbbbbbbbbbbbbbbb" << endl;
            cout << "bbbbbbbbbbbbbbbbbbbbbbbbbbb" << endl;
            cout << "bbbbbbbbbbbbbbbbbbbbbbbbbbb" << endl;
            cout << "bbbbbbbbbbbbbbbbbbbbbbbbbbb" << endl;
            cout << "bbbbbbbbbbbbbbbbbbbbbbbbbbb" << endl;
            cout << "bbbbbbbbbbbbbbbbbbbbbbbbbbb" << endl;
            color = 2;
        }

        vector<int> dominant_move_indices;
        vector<int> not_dominant_move_indices;
        shared_ptr<Node> parent_shared_ptr;
        // Check if parent is root
        bool parent_node_is_root = false;
        if (!this->is_root) {
            parent_shared_ptr = parent.lock();
            parent_node_is_root = parent_shared_ptr->is_root;
        }

        cout << "ggggggggggggggggggggg" << endl;
        cout << "ggggggggggggggggggggg" << endl;
        cout << "ggggggggggggggggggggg" << endl;
        cout << "ggggggggggggggggggggg" << endl;
        cout << "ggggggggggggggggggggg" << endl;

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
            temp_move.color = color;
            vector<Movement> movements;
            movements.clear();
            movements.push_back(temp_move);
            cout << endl << endl;
            cout << "qqqqqqqqqqqqqqqq" << endl;
            cout << "qqqqqqqqqqqqqqqq" << endl;
            cout << "qqqqqqqqqqqqqqqq" << endl;
            cout << "qqqqqqqqqqqqqqqq" << endl;
            cout << "qqqqqqqqqqqqqqqq" << endl;
            Properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
            ///////////////////////////////////////////////////////
            // DOMINANT PRUNING
            ///////////////////////////////////////////////////////
            // If use dominant pruning, check children's properties.
            // If child move lead to a dominant strategy (We use "GETTING A NEW LINE")
            // Thus, the move that can get a new line we call it "dominant", and its parent move we call it "dominated".
            // Rules:
            //     From non-root node to find move:
            //         If a move is "dominant":
            //                             1. Prune its parent move
            //                             2. Check if the parent move origin node has no child because of your pruning,
            //                                If it is, make that node's is_terminated to be TRUE.
            //     From root node to find move:
            //         1. Expand all children and start to check it.
            //         2. If there are ANY "dominant" moves,
            //                     prune all "non-dominant" moves (including regular and dominated moves),
            //                     (Actually, A move will be only dominant or regular here;
            //            else (there are only regular moves) (You cannot know if your move is dominated unless you know the next layer),
            //                     do nothing
            //      If we need to prune the move from root node, we don't directly prune it,
            //                     we set the node which is produced from root node and the move to be NEGATIVE INFINITE.
            //                     Thus, that node will not be visited anymore.
            // P.S. A move pruned means its following node is pruned.

            if (dominant_pruning) {
                // Check if it is a dominant move or a regular move.
                bool is_dominant = false;
                if ((color == BLACK_PLAYER_COLOR && (new_properties.black_lines - my_properties.black_lines > 0))
                    || (color = WHITE_PLAYER_COLOR && (new_properties.white_lines - my_properties.white_lines > 0))) {
                    is_dominant = true;
                }
                //
                if (is_dominant) {
                    if (!is_root) { // Non-root node
                        if (parent_node_is_root) {
                            this->is_pruned = true;
                        } else { // Not affect first layer
                            // Check if parent will have no children after this child being pruned
                            if (parent_shared_ptr->children.size() == 1) {
                                // If so, make parent node be terminated
                                parent_shared_ptr->is_terminated = true;
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
                        }
                        return EXPAND_PRUNE_PARENT;
                    } else { // Root node
                        dominant_move_indices.emplace_back(legal_move_idx);
                    }
                } else {
                    if (is_root) {
                        not_dominant_move_indices.emplace_back(legal_move_idx);
                    }
                }
            }
            temp_board[temp_move.l][temp_move.x][temp_move.y] = color;
            shared_ptr<Node> new_child = make_shared<Node>(temp_board, this->hands + 1, temp_move, new_properties);
            this->children.push_back(new_child);
            new_child->parent = shared_from_this();
        }

        if (this->is_root && dominant_pruning && (!dominant_move_indices.empty())) {
            // Prune all non-dominant
            for (int idx: not_dominant_move_indices) {
                this->children[idx]->is_pruned = true;
            }
            return EXPAND_NOT_PRUNE_PARENT;
        }

        if (dominant_pruning && parent_node_is_root) {
            // Check if parent has no child after being pruned
            bool has_child = false;
            for (auto &child: parent_shared_ptr->children) {
                if (!child->is_pruned) {
                    has_child = true;
                    break;
                }
            }
            if (!has_child) {
                return EXPAND_ROOT_HAS_NO_CHILD_AFTER_PRUNED;
            }
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
        for (int i = hands; i < 64; i++) {
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
        cout << "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" << endl;
        cout << "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" << endl;
        cout << "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" << endl;
        cout << "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" << endl;
        cout << "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" << endl;
        cout << "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz" << endl;
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
        vector<Movement> block_moves;
        for (auto &all_possible_move : all_possible_moves) {
            for (auto &dir : dirs) {
                unique_ptr<int[]> new_pos_a(new int[3]);
                new_pos_a[0] = all_possible_move.l + dir[0];
                new_pos_a[1] = all_possible_move.x + dir[1];
                new_pos_a[2] = all_possible_move.y + dir[2];

                unique_ptr<int[]> new_pos_b(new int[3]);
                new_pos_b[0] = all_possible_move.l + dir[0];
                new_pos_b[1] = all_possible_move.x + dir[1];
                new_pos_b[2] = all_possible_move.y + dir[2];
                int l = new_pos_b[0], i = new_pos_b[1], j = new_pos_b[2];
                if (boundary_test(new_pos_b) && this->board[l][i][j] == self_color) {
                    new_pos_b[0] += dir[0];
                    new_pos_b[1] += dir[1];
                    new_pos_b[2] += dir[2];
                    l = new_pos_b[0];
                    i = new_pos_b[1];
                    j = new_pos_b[2];
                    if (boundary_test(new_pos_b) && this->board[l][i][j] == self_color) {
                        int k = 10;
                        while (k > 0) {
                            Movement block_move_b;
                            block_move_b.l = all_possible_move.l;
                            block_move_b.x = all_possible_move.x;
                            block_move_b.y = all_possible_move.y;
                            block_move_b.color = self_color;
                            block_moves.push_back(block_move_b);
                            k--;
                        }
                    }
                }

                l = new_pos_a[0];
                i = new_pos_a[1];
                j = new_pos_a[2];
                if (!boundary_test(new_pos_a) || this->board[l][i][j] != opponent_color)
                    continue;
                new_pos_a[0] += dir[0];
                new_pos_a[1] += dir[1];
                new_pos_a[2] += dir[2];
                l = new_pos_a[0];
                i = new_pos_a[1];
                j = new_pos_a[2];
                if (!boundary_test(new_pos_a) || this->board[l][i][j] != opponent_color)
                    continue;
                Movement block_move_a;
                block_move_a.l = all_possible_move.l;
                block_move_a.x = all_possible_move.x;
                block_move_a.y = all_possible_move.y;
                block_move_a.color = self_color;
                block_moves.push_back(block_move_a);
            }
        }
        srand(time(nullptr));
        if (!block_moves.empty()) {
            return block_moves;
        } else {
            return all_possible_moves;
        }
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

//        cout << "MMMMMMMMMMMMMMM ";
//        this->my_properties.print_properties();
//        movements[0].print_movement();
        cout << "xxxxxxxxxxxxxxx" << endl;
        cout << "xxxxxxxxxxxxxxx" << endl;
        cout << "xxxxxxxxxxxxxxx" << endl;
        cout << "xxxxxxxxxxxxxxx" << endl;
        cout << "xxxxxxxxxxxxxxx" << endl;
        Properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
//        cout << "^^^^^^^^^^^^^^" << endl;
//        new_properties.print_properties();
//        cout << "^^^^^^^^^^^^^^" << endl;
//        cout << endl;
        temp_board[next_move.l][next_move.x][next_move.y] = color;
        shared_ptr<Node> ret = make_shared<Node>(temp_board, this->hands + 1, next_move, new_properties);
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

private:
    vector<vector<float>> first_layer_stat(int mode, const string &output_filename_suffix) {
        // If output_filename_suffix == empty string means:
        //     "not to output to file"
        // Mode
        //     1: visiting_count
        //     2: value_sum
        //     3: visiting_count / value_sum
        vector<vector<float>> plane(6);
        string mode_name;
        for (auto &row: plane) {
            row.resize(6);
        }
        for (auto &child: root->children) {
            float value;
            switch (mode) {
                case 1:
                    value = (float) child->visiting_count;
                    mode_name = "Visiting_Count";
                    break;
                case 2:
                    value = (float) child->value_sum;
                    mode_name = "Value_Sum";
                    break;
                case 3:
                    value = (float) child->value_sum / (float) child->visiting_count;
                    mode_name = "Value_Mean";
                    break;
                default:
                    cerr << "Invalid mode: " << mode << endl;
                    exit(EXIT_FAILURE);
            }
            plane[child->move.x][child->move.y] = value;
        }
        // Output file
        if (!output_filename_suffix.empty()) {
            ofstream ofs("first_layer_stat_" + output_filename_suffix + ".txt", fstream::trunc);
            ofs << mode_name;
            for (const auto &line: plane) {
                for (auto item: line) {
                    ofs << " " << item;
                }
            }
            ofs.close();
        }
        // Return 2D vector
        return plane;
    }

public:
    shared_ptr<Node> root;
    int cur_simulation_cnt;
    int max_simulation_cnt;
    int max_time_sec;
    bool print_simulation_cnt;

    MCTS(shared_ptr<Node> root, int max_simulation_cnt, int max_time_sec, bool print_simulation_cnt) {
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
                 bool dominate_pruning) {

        // If it is first hand (black)
        if (first_hand_center && root->hands == 0) {
            return get_rand_first_hand_center_move(false);
        }


        this->cur_simulation_cnt = 0;
        this->root->is_root = true;

        // Time example:
        // auto t0 = Time::now();
        // auto t1 = Time::now();
        // sec duration = t1 - t0;
        // std::cout << duration.count() << "s\n";

        // clock_t start = clock();
        auto start = Time::now();
        bool root_has_no_child_since_all_are_pruned = false;
        while (true) {
            /////////////////////////////////////////////////////
            // SELECT
            /////////////////////////////////////////////////////
            shared_ptr<Node> temp_node = this->root->select();
            // Check if selected node is terminated (e.g. hands == 64 or other factors)
            if (temp_node->hands >= 64) { // Number of hands reaches limit.
                temp_node->is_terminated = true;
            }
            /////////////////////////////////////////////////////
            // EXPAND
            /////////////////////////////////////////////////////
            if (!temp_node->is_terminated) {
                cout << "oooooooooooooooooo" << endl;
                cout << "oooooooooooooooooo" << endl;
                cout << "oooooooooooooooooo" << endl;
                cout << "oooooooooooooooooo" << endl;
                int expand_rt = temp_node->expand(dominate_pruning);
                // expand_rt==EXPAND_NOT_PRUNE_PARENT is ok
                if (expand_rt == EXPAND_PRUNE_PARENT) {
                    continue; // Drop this simulation
                } else if (expand_rt == EXPAND_ROOT_HAS_NO_CHILD_AFTER_PRUNED) {
                    root_has_no_child_since_all_are_pruned = true;
                    break;
                }
            }
            ///////////////////////////////////////////////////
            // EVALUATE
            ///////////////////////////////////////////////////
            cout << "pppppppppppppppppppppppp" << endl;
            cout << "pppppppppppppppppppppppp" << endl;
            cout << "pppppppppppppppppppppppp" << endl;
            cout << "pppppppppppppppppppppppp" << endl;
            cout << "pppppppppppppppppppppppp" << endl;
            int reward = temp_node->playout(playout_use_block_move);
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
            auto end = Time::now();
            sec duration = Time::now() - start;
            if (duration.count() >= (double) (this->max_time_sec))
                break;
            if (this->cur_simulation_cnt >= this->max_simulation_cnt)
                break;
        }

        if (this->print_simulation_cnt) {
            cout << "Simulation cnt: " << this->cur_simulation_cnt << endl;
        }

        /////////////////////////////////////////////////
        // PICK A MOVE
        /////////////////////////////////////////////////

        // Root no child. We random pick a child.
        if (root_has_no_child_since_all_are_pruned) {
            auto moves = this->root->get_next_possible_move();
            int rand_num = rand() % moves.size();
            return moves[rand_num];
        }

        //
        double winning_rate = NEG_INFINITE;
        Movement ret;
        for (const auto &temp_node : this->root->children) {
            if (temp_node->visiting_count == 0)
                continue;
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
            Movement move = mcts.run(false, true, true, false);
            shared_ptr<Node> new_node = cur_node->get_node_after_playing(move);
            cur_node = new_node;
        }
        cout << " (finished) " << endl;
        return cur_node;
    }

    ~MCTS() {
        root.reset();
    }


    vector<vector<float>> first_layer_visit_cnt_distribution(const string &output_filename_suffix) {
        // Output filename == empty string means "not to output to file"
        return first_layer_stat(1, output_filename_suffix);
    }

    vector<vector<float>> first_layer_value_sum_distribution(const string &output_filename_suffix) {
        // Output filename == empty string means "not to output to file"

        return first_layer_stat(2, output_filename_suffix);
    }

    vector<vector<float>> first_layer_value_mean_distribution(const string &output_filename_suffix) {
        // Output filename == empty string means "not to output to file"
        return first_layer_stat(3, output_filename_suffix);
    }

};

int main() {
    // Init random
    srand(time(nullptr));

//    shared_ptr<Node> node = MCTS::get_init_node();
//    vector<Movement> moves;
//    node = node->get_node_after_playing(Movement(0, 2, 1, 2)
//    );
//    node = node->get_node_after_playing(Movement(1, 1, 1, 1)
//    );
//    node = node->get_node_after_playing(Movement(1, 2, 1, 2)
//    );
//    node = node->get_node_after_playing(Movement(2, 1, 1, 1)
//    );
//    node = node->get_node_after_playing(Movement(2, 2, 1, 2)
//    );
//    node = node->get_node_after_playing(Movement(3, 1, 1, 1)
//    );
//    node = node->get_node_after_playing(Movement(3, 2, 1, 2)
//    );
//    node = node->get_node_after_playing(Movement(4, 1, 1, 1)
//    );
//    node = node->get_node_after_playing(Movement(4, 2, 1, 2)
//    );
//    node = node->get_node_after_playing(Movement(5, 1, 1, 1)
//    );
//    moves.emplace_back(0, 1, 1, 1);
//    moves.emplace_back(0, 2, 1, 2);
//    moves.emplace_back(1, 1, 1, 1);
//    moves.emplace_back(1, 2, 1, 2);
//    moves.emplace_back(2, 1, 1, 1);
//    moves.emplace_back(2, 2, 1, 2);
//    moves.emplace_back(3, 1, 1, 1);
//    moves.emplace_back(3, 2, 1, 2);
//    moves.emplace_back(4, 1, 1, 1);
//    moves.emplace_back(4, 2, 1, 2);
//
//    for (auto move: moves) {
//        node = node->get_node_after_playing(move);
//    }
//
//    Properties prop = get_state_properties_b(MCTS::get_init_node()->board, MCTS::get_init_node()->my_properties, moves);
//
//    cout << prop.black_points << " " << prop.white_points << " " << prop.black_lines << " " << prop.white_lines << endl;
//    node->output_board_string_for_plot_state();
//    system("python3 plot_state.py board_content_for_plotting.txt");
//
//    exit(1);

    // Fight
    string fight_record_dir = "fight_dir";
    int max_simulation_cnt = 999999;
    int max_simulation_time = 20;
    bool plot_state_instantly = false;
    shared_ptr<Node> cur_node = MCTS::get_init_node();
    for (int i = 0; i < 64; i += 2) {
        cout << "================ i = " << i << " =================" << endl;
        Movement move;
        string output_path;
        Properties prop;

        // Black's turn
        MCTS mcts_black(cur_node, max_simulation_cnt, max_simulation_time, true);
        move = mcts_black.run(false, false, true, false);
        cur_node = cur_node->get_node_after_playing(move);
        cur_node->my_properties.print_properties();

        //
        cur_node->output_board_string_for_plot_state();
        cur_node->my_properties.output_properties();
        output_path = fight_record_dir + "/hands_" + to_string(i + 1) + "_blackDone.png";
        if (plot_state_instantly)
            system(string("python3 plot_state_and_output.py board_content_for_plotting.txt " + output_path +
                          " output_properties_for_plotting.txt").c_str());
        ///////////////////////////////
        // Plot 2d plane
        ///////////////////////////////
        print_vector_2d_plane(mcts_black.first_layer_value_sum_distribution("valueSum"));
        print_vector_2d_plane(mcts_black.first_layer_visit_cnt_distribution("visitCnt"));
        print_vector_2d_plane(mcts_black.first_layer_value_mean_distribution("valueMean"));
        cout << "Get move: ";
        move.print_movement();


        // White's turn
        MCTS mcts_white(cur_node, max_simulation_cnt, max_simulation_time, true);
        move = mcts_white.run(false, false, true, true);
        cur_node = cur_node->get_node_after_playing(move);
        cur_node->my_properties.print_properties();



        //
        cur_node->output_board_string_for_plot_state();
        cur_node->my_properties.output_properties();
        output_path = fight_record_dir + "/hands_" + to_string(i + 2) + "_whiteDone.png";
        if (plot_state_instantly)
            system(string("python3 plot_state_and_output.py board_content_for_plotting.txt " + output_path +
                          " output_properties_for_plotting.txt").c_str());
        ///////////////////////////////
        // Plot 2d plane
        ///////////////////////////////
        print_vector_2d_plane(mcts_white.first_layer_value_sum_distribution("valueSum"));
        print_vector_2d_plane(mcts_white.first_layer_visit_cnt_distribution("visitCnt"));
        print_vector_2d_plane(mcts_white.first_layer_value_mean_distribution("valueMean"));
        cout << "Get move: ";
        move.print_movement();


        cout << endl;

    }

    ///////////////////////////////
    // Plot 2d plane
    ///////////////////////////////
    //    print_vector_2d_plane(mcts.first_layer_value_sum_distribution("valueSum"));
    //    print_vector_2d_plane(mcts.first_layer_visit_cnt_distribution("visitCnt"));
    //    print_vector_2d_plane(mcts.first_layer_value_mean_distribution("valueMean"));

    ///////////////////////////////
    // Test gen_block_move()
    ///////////////////////////////
    //    vector<Movement> v = cur_node->gen_block_move();
    //    for (auto &a_move: v) {
    //        cout << a_move.l << ", " << a_move.x << ", " << a_move.y << endl;
    //    }
    //    cout << "Total: " << v.size() << " moves." << endl;

    //////////////////////////
    // Plot state
    //////////////////////////
    //    cur_node->output_board_string_for_plot_state();
    //    system("python3 plot_state.py board_content_for_plotting.txt");

    ////////////////////////////
    // Print movements
    ////////////////////////////
    //    for (auto item : movements) {
    //        item.print_movement();
    //    }
}
