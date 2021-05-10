#ifndef MASTERTHEGAMEOFCONNECT4_NODE_H
#define MASTERTHEGAMEOFCONNECT4_NODE_H

#include "Movement.h"
#include "Properties.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>
#include <array>

typedef std::array<std::array<std::array<int, 6>, 6>, 6> array3d_int;

class Node : public std::enable_shared_from_this<Node> {
public:
    array3d_int board{};
    int hands;
    Movement move;
    int visiting_count;
    int value_sum;
    bool is_leaf;
    bool is_root;
    bool is_terminated; // Prevent from doing expanding
    std::vector<std::shared_ptr<Node>> children;
    Properties my_properties;
    std::weak_ptr<Node> parent;

    static Properties get_state_properties_b(array3d_int start_state, Properties start_state_properties,
                                             const std::vector<Movement> &movements);

    static bool boundary_test(const std::shared_ptr<int[]> &coordinate);

    Node(array3d_int board, int hands, Movement move, Properties my_properties);

    ~Node();

    static double ucb(int child_visiting_count, int winning_count, double parent_visit_cnt);

    // PUCT: Predictor + UCB applied to trees
    // TODO: Make it easy to tune the hyper-parameters
    static double puct(int child_visiting_count, int winning_count, double parent_visit_cnt, double prior_value);

    std::shared_ptr<Node> select(bool prior_value);

    int expand(bool dominant_pruning, bool prior_value, bool not_check_dominant_on_first_layer);

    int playout(bool use_block_moves);

    void backup(int reward, bool use_reverse_for_opponent);

    std::vector<Movement> get_next_possible_move();

    std::vector<Movement> gen_block_move();

    std::shared_ptr<Node> get_node_after_playing(Movement next_move) const;

    void output_board_string_for_plot_state();

    void print_flattened_board();
};

#endif //MASTERTHEGAMEOFCONNECT4_NODE_H
