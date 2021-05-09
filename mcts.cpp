#include "Node.h"
#include "MCTS.h"
#include "Movement.h"
#include "Properties.h"
#include "Engine.h"
#include "tools.h"

#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <array>
#include <algorithm>
#include <iomanip>

using namespace std;

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

    for (int iii = 0; iii < 99999; iii++) {    // Fight
        string fight_record_dir = "fight_dir";
        int max_simulation_cnt = 9999999;
        double max_simulation_time = 4;
        bool plot_state_instantly = false;
        shared_ptr<Node> cur_node = MCTS::get_init_node();
        bool debug = true;

//    Movement m1(0, 2, 1, 2);
//    Movement m11(0, 3, 1, 1);
//    Movement m2(0, 2, 2, 2);
//    Movement m22(0, 3, 2, 1);
//    Movement m3(0, 2, 3, 2);
//    Movement m33(1, 2, 1, 1);
//    cur_node = cur_node->get_node_after_playing(m11);
//    cur_node = cur_node->get_node_after_playing(m1);
//
//    cur_node = cur_node->get_node_after_playing(m22);
//    cur_node = cur_node->get_node_after_playing(m2);
//
//    cur_node = cur_node->get_node_after_playing(m33);
//    cur_node = cur_node->get_node_after_playing(m3);


        for (int i = cur_node->hands; i < 64; i += 2) {
            cout << "================ i = " << i << " =================" << endl;
            Movement move;
            string output_path;
            Properties prop;

            // Black's turn
            cout << "########### BLACK #####################" << endl;
            MCTS mcts_black(cur_node, max_simulation_cnt, max_simulation_time, true);
            move = mcts_black.run(false, false, true, true, true, false);
            cur_node = cur_node->get_node_after_playing(move);
            if (debug) { cur_node->my_properties.print_properties(); }

            //
            if (debug) {
                cur_node->output_board_string_for_plot_state();
                cur_node->my_properties.output_properties();
                output_path = fight_record_dir + "/hands_" + to_string(i + 1) + "_blackDone.png";
                if (plot_state_instantly)
                    system(string("python3 plot_state_and_output.py board_content_for_plotting.txt " + output_path +
                                  " output_properties_for_plotting.txt").c_str());
            }
            ///////////////////////////////
            // Plot 2d plane
            ///////////////////////////////
            if (debug) {
                tools::print_vector_2d_plane(mcts_black.first_layer_value_sum_distribution("valueSum"));
                tools::print_vector_2d_plane(mcts_black.first_layer_visit_cnt_distribution("visitCnt"));
                tools::print_vector_2d_plane(mcts_black.first_layer_value_mean_distribution("valueMean"));
                cout << "Get move: ";
                move.print_movement();
            }


            // White's turn
            cout << "########### WHITE #####################" << endl;
            MCTS mcts_white(cur_node, max_simulation_cnt, max_simulation_time, true);
            move = mcts_white.run(false, false, true, true, true, false);
            cur_node = cur_node->get_node_after_playing(move);
            if (debug) { cur_node->my_properties.print_properties(); }



            //
            if (debug) {
                cur_node->output_board_string_for_plot_state();
                cur_node->my_properties.output_properties();
                output_path = fight_record_dir + "/hands_" + to_string(i + 2) + "_whiteDone.png";
                if (plot_state_instantly)
                    system(string("python3 plot_state_and_output.py board_content_for_plotting.txt " + output_path +
                                  " output_properties_for_plotting.txt").c_str());
            }
            ///////////////////////////////
            // Plot 2d plane
            ///////////////////////////////
            if (debug) {
                tools::print_vector_2d_plane(mcts_white.first_layer_value_sum_distribution("valueSum"));
                tools::print_vector_2d_plane(mcts_white.first_layer_visit_cnt_distribution("visitCnt"));
                tools::print_vector_2d_plane(mcts_white.first_layer_value_mean_distribution("valueMean"));
                cout << "Get move: ";
                move.print_movement();
            }


            cout << endl;

        }
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
