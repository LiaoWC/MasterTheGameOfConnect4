//
// Created by smallfish on 5/10/21.
//

#include "Node.h"
#include "Engine.h"


bool Node::boundary_test(const std::unique_ptr<int[]> &coordinate) {
    for (int i = 0; i < 3; i++)
        if (coordinate[i] < 0 || coordinate[i] >= 6)
            return false;
    return true;
}


Node::Node(int board[6][6][6],
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

Node::~Node() {
    children.clear();
}

double Node::ucb(int child_visiting_count,
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
double Node::puct(int child_visiting_count,
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

std::shared_ptr<Node> Node::select(bool prior_value) {
    std::shared_ptr<Node> cur = shared_from_this();
    while (true) {
        if (cur->is_leaf)
            return cur;
        else {
            // double cur_max_ucb_result = -1;
            double cur_max_ucb_result = NEG_INFINITE;
            std::vector<double> ucb_results;
            for (unsigned int i = 0; i < cur->children.size(); i++) {
                std::shared_ptr<Node> temp_node = cur->children[i];
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
            std::vector<unsigned int> max_value_idx;
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
            std::shared_ptr<Node> next_node = cur->children[max_value_idx[random_number]];
            cur = next_node;
        }
    }
}

int Node::expand(bool dominant_pruning, bool prior_value, bool not_check_dominant_on_first_layer) {
    this->is_leaf = false;
    int temp_board[6][6][6];
    std::vector<Movement> legal_moves;
    if (prior_value) {
        legal_moves = this->gen_block_move();
    } else {
        legal_moves = this->get_next_possible_move();
    }
    std::vector<int> dominant_move_indices;
    std::shared_ptr<Node> parent_shared_ptr;
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
        std::vector<Movement> movements;
        movements.clear();
        movements.push_back(temp_move);


        Properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
        temp_board[temp_move.l][temp_move.x][temp_move.y] = color;

        ///////////////////////////////////////////////////////
        // DOMINANT PRUNING
        ///////////////////////////////////////////////////////
        // If use dominant pruning, check children's properties.
        // If child move lead to a dominant strategy (We use "GETTING A NEW LINE")
        // Thus, the move that can get a new line we call it "dominant", and its parent move we call it "dominated".
        // Rules:
        //     From non-root node to find move:
        //         If a move is "dominant":
        //                             1. Prune its parent move.
        //                             2. Check if the parent move origin node has no child because of your pruning,
        //                                If it is, make that node's is_terminated and is_leaf to be TRUE.
        //     From root node to find move:
        //         1. Expand all children and start to check it.
        //         2. If there are ANY "dominant" moves,
        //                     prune all "non-dominant" moves (including regular and dominated moves),
        //                     (Actually, A move will be only dominant or regular here;
        //            else (there are only regular moves) (You cannot know if your move is dominated unless you know the next layer),
        //                     do nothing
        // P.S. A move pruned means its following node is pruned.
        //
        // 2021/5/9 Modification:
        //                       - Root's children will be erased from std::vector if pruned.
        //                       - If root's children all are pruned, use smaller mcts to random again.
        //                             In run(), we check if root has no children anymore. If it is,
        //                             we use the rest of time to do a new mcts run without pruning.
        // P.S. We must ensure first layer children get at least one visit
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
                    return EXPAND_PRUNE_PARENT;
                } else { // Root node
                    dominant_move_indices.emplace_back(legal_move_idx);
                }
            }
        }
        std::shared_ptr<Node> new_child = std::make_shared<Node>(temp_board, this->hands + 1, temp_move,
                                                                 new_properties);
        this->children.push_back(new_child);
        new_child->parent = shared_from_this();
    }

    // For root node: If there are any dominant moves existing, we prune all non-dominant nodes.
    if (this->is_root && dominant_pruning && (!dominant_move_indices.empty())) {
        std::vector<std::shared_ptr<Node>> children_pruned;
        // "Better performance will be reached when avoiding dynamic reallocation, so try to have the std::vector memory be big enough to receive all elements."
        //  -- reference: https://stackoverflow.com/questions/32199388/what-is-better-reserve-std::vector-capacity-preallocate-to-size-or-push-back-in-loo
        children_pruned.reserve(dominant_move_indices.size());
        for (int idx: dominant_move_indices) {
            children_pruned.emplace_back(this->children[idx]);
        }
        this->children = children_pruned;
        return EXPAND_ROOT_HAS_DOMINANT_MOVE;
    }
    return EXPAND_NOT_PRUNE_PARENT;
}

int Node::playout(bool use_block_moves) {
    int temp_board[6][6][6];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                temp_board[i][j][k] = this->board[i][j][k];
            }
        }
    }
    std::vector<Movement> movements;
    movements.clear();
    for (int i = this->hands; i < 64; i++) {
        std::vector<Movement> considered_moves;
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

void Node::backup(int reward, bool use_reverse_for_opponent) {
    std::shared_ptr<Node> cur = shared_from_this();
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

std::vector<Movement> Node::get_next_possible_move() {
    std::vector<Movement> ret;
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

std::vector<Movement> Node::gen_block_move() {
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
    std::vector<Movement> all_possible_moves = this->get_next_possible_move();
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
    std::vector<Movement> block_moves;
    for (auto &all_possible_move : all_possible_moves) {
        double temp_p = 0;
        flag = true;
        flag2 = true;
        for (auto &dir : dirs) {
            std::unique_ptr<int[]> new_pos(new int[3]);
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
                std::unique_ptr<int[]> new_pos(new int[3]);
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
                std::unique_ptr<int[]> new_pos(new int[3]);
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

std::shared_ptr<Node> Node::get_node_after_playing(Movement next_move) {
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
    std::vector<Movement> movements;
    movements.clear();
    movements.push_back(next_move);

    Properties new_properties = get_state_properties_b(temp_board, this->my_properties, movements);
    temp_board[next_move.l][next_move.x][next_move.y] = color;
    std::shared_ptr<Node> ret = std::make_shared<Node>(temp_board, this->hands + 1, next_move, new_properties);
    return ret;
}

void Node::output_board_string_for_plot_state() {
    std::string board_str;
    for (auto &d : board) {
        for (auto &i : d) {
            for (int j : i) {
                board_str += " " + std::to_string(j);
            }
        }
    }
    board_str += "\n";
    std::string output_filename = "board_content_for_plotting.txt";
    std::ofstream ofs(output_filename, std::fstream::trunc);
    ofs << board_str;
    ofs.close();
    std::cout << "Output board std::string to \"" << output_filename << "\" done!" << std::endl;
}

void Node::print_flattened_board() {
    for (auto &l : this->board) {
        for (auto &x : l) {
            for (int y : x) {
                std::cout << y << " ";
            }
        }
    }
    std::cout << std::endl;
}
