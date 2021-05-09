#ifndef MASTERTHEGAMEOFCONNECT4_NODE_H
#define MASTERTHEGAMEOFCONNECT4_NODE_H
#include "Movement.h"
#include "Properties.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>

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
    std::vector<std::shared_ptr<Node>> children;
    Properties my_properties;
    std::weak_ptr<Node> parent;

    static Properties get_state_properties_b(int start_state[6][6][6],
                                             Properties start_state_properties,
                                             const std::vector<Movement> &movements) {
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
                    std::unique_ptr<int[]> temp_coordinate(new int[3]);
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
                    std::unique_ptr<int[]> temp_coordinate(new int[3]);
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
                        temp_properties.black_points +=
                                100 / (temp_properties.black_lines + temp_properties.white_lines);
                    } else {
                        temp_properties.white_lines += 1;
                        temp_properties.white_points +=
                                100 / (temp_properties.black_lines + temp_properties.white_lines);
                    }
                    cnt -= 1;
                }
            }
            temp_state[l][i][j] = c;
        }
        return temp_properties;
    }


    static bool boundary_test(const std::unique_ptr<int[]> &coordinate);

    Node(int board[6][6][6], int hands, Movement move, Properties my_properties);

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

    std::vector<Movement> gen_block_move() ;

    std::shared_ptr<Node> get_node_after_playing(Movement next_move);

    void output_board_string_for_plot_state();

    void print_flattened_board();
};


#endif //MASTERTHEGAMEOFCONNECT4_NODE_H
