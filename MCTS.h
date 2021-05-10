#ifndef MASTERTHEGAMEOFCONNECT4_MCTS_H
#define MASTERTHEGAMEOFCONNECT4_MCTS_H

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>

// Forward
class Node;
class Movement;
class Properties;


class MCTS {

private:
    std::vector<std::vector<float>> first_layer_stat(int mode, const std::string &output_filename_suffix);

public:
    std::shared_ptr<Node> root;
    int cur_simulation_cnt{};
    int max_simulation_cnt{};
    double max_time_sec{};
    bool print_simulation_cnt{};

    MCTS()= default;
    MCTS(std::shared_ptr<Node> root, int max_simulation_cnt, double max_time_sec, bool print_simulation_cnt);

    static Movement get_rand_first_hand_center_move(bool random);

    Movement run(bool playout_use_block_move,
                 bool reward_add_score_diff,
                 bool first_hand_center,
                 bool dominate_pruning,
                 bool prior_value,
                 bool not_check_dominate_pruning_on_first_layer);

    static std::shared_ptr<Node> get_init_node();

    static std::shared_ptr<Node> get_random_board_node(int step, int max_simulation_cnt, int max_simulation_time);

    ~MCTS();

    std::vector<std::vector<float>> first_layer_visit_cnt_distribution(const std::string &output_filename_suffix) ;

    std::vector<std::vector<float>> first_layer_value_sum_distribution(const std::string &output_filename_suffix);

    std::vector<std::vector<float>> first_layer_value_mean_distribution(const std::string &output_filename_suffix) ;
};

#endif //MASTERTHEGAMEOFCONNECT4_MCTS_H
