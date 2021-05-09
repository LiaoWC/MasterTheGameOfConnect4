#include "MCTS.h"
#include "Node.h"
#include "Engine.h"
#include <chrono>
#include <memory>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double> sec;


std::vector<std::vector<float>> MCTS::first_layer_stat(int mode, const std::string &output_filename_suffix) {
    // If output_filename_suffix == empty std::string means:
    //     "not to output to file"
    // Mode
    //     1: visiting_count
    //     2: value_sum
    //     3: visiting_count / value_sum
    std::vector<std::vector<float>> plane(6);
    std::string mode_name;
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
                std::cerr << "Invalid mode: " << mode << std::endl;
                exit(EXIT_FAILURE);
        }
        plane[child->move.x][child->move.y] = value;
    }
    // Output file
    if (!output_filename_suffix.empty()) {
        std::ofstream ofs("first_layer_stat_" + output_filename_suffix + ".txt", std::fstream::trunc);
        ofs << mode_name;
        for (const auto &line: plane) {
            for (auto item: line) {
                ofs << " " << item;
            }
        }
        ofs.close();
    }
    // Return 2D std::vector
    return plane;
}

MCTS::MCTS(std::shared_ptr<Node> root, int max_simulation_cnt, double max_time_sec, bool print_simulation_cnt) {
    this->root = std::move(root);
    this->max_simulation_cnt = max_simulation_cnt;
    this->max_time_sec = max_time_sec;
    this->cur_simulation_cnt = 0;
    this->print_simulation_cnt = print_simulation_cnt;
}

Movement MCTS::get_rand_first_hand_center_move(bool random) {
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

Movement MCTS::run(bool playout_use_block_move,
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
    // std::std::cout << duration.count() << "s\n";

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
        std::shared_ptr<Node> temp_node = this->root->select(prior_value);
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
        //            std::cout << 4 << std::endl;
        temp_node->backup(reward, true);
        this->cur_simulation_cnt++;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Re run MCTS if root's children are all pruned but not because root has any dominant moves
    /////////////////////////////////////////////////////////////////////////////////////////////
    if (!root_has_dominant_move && this->root->is_terminated) {
        std::shared_ptr<Node> new_root = std::make_shared<Node>(this->root->board, this->root->hands, this->root->move,
                                                                this->root->my_properties);
        auto end = Time::now();
        sec duration = Time::now() - start;
        std::cout << "Has used " << duration.count() << " sec" << std::endl;
        double rest_of_time = ((double) (this->max_time_sec) - duration.count());
        double rerun_time = std::max(1., rest_of_time);
        MCTS mcts(new_root, 999999, rerun_time, true);
        std::cout << "Re-mcts-run with remained time: " << rerun_time << " sec." << std::endl;
        // Run mcts with simple settings
        std::cout << new_root->hands << " DDDDDDDDDDDDD" << std::endl;
        Movement new_move = mcts.run(false, false, false, false, false, false);
        this->root = new_root;
        return new_move;
    }

    if (this->print_simulation_cnt) {
        std::cout << "Simulation cnt: " << this->cur_simulation_cnt << std::endl;
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
        //std::cout << temp_node->value_sum << " " << temp_node->visiting_count << std::endl;
        double value_mean = (double) temp_node->value_sum / (double) temp_node->visiting_count;
        //std::cout << temp_winning_rate << std::endl;
        if (value_mean > winning_rate) {
            winning_rate = value_mean;
            ret = temp_node->move;
            //std::cout << temp_node->move.l << " " << temp_node->move.x << " " << temp_node->move.y << std::endl;
        }
    }
    return ret;
}

std::shared_ptr<Node> MCTS::get_init_node() {
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
    std::shared_ptr<Node> start_node = std::make_shared<Node>(b, 0, move, start_properties);
    return start_node;
}

std::shared_ptr<Node> MCTS::get_random_board_node(int step, int max_simulation_cnt, int max_simulation_time) {
    std::shared_ptr<Node> cur_node = MCTS::get_init_node();
    std::cout << "Getting random board..." << std::endl;
    std::cout << "Move done:" << std::endl;
    for (int i = 0; i < step; i++) {
        MCTS mcts(cur_node, max_simulation_cnt, max_simulation_time, true);
        std::cout << i + 1 << " " << std::flush;
        Movement move = mcts.run(false, false, false, false, false, false);
        std::shared_ptr<Node> new_node = cur_node->get_node_after_playing(move);
        cur_node = new_node;
    }
    std::cout << " (finished) " << std::endl;
    return cur_node;
}

MCTS::~MCTS() {
    root.reset();
}

std::vector<std::vector<float>> MCTS::first_layer_visit_cnt_distribution(const std::string &output_filename_suffix) {
    // Output filename == empty std::string means "not to output to file"
    return first_layer_stat(1, output_filename_suffix);
}

std::vector<std::vector<float>> MCTS::first_layer_value_sum_distribution(const std::string &output_filename_suffix) {
    // Output filename == empty std::string means "not to output to file"

    return first_layer_stat(2, output_filename_suffix);
}

std::vector<std::vector<float>> MCTS::first_layer_value_mean_distribution(const std::string &output_filename_suffix) {
    // Output filename == empty std::string means "not to output to file"
    return first_layer_stat(3, output_filename_suffix);
}



