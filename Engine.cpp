#include "Engine.h"

using namespace std;

int main() {
    // Init random
    srand(time(nullptr));

    for (int iii = 0; iii < 99999; iii++) {    // Fight
        string fight_record_dir = "fight_dir";
        int max_simulation_cnt = 9999999;
        double max_simulation_time = 4;
        bool plot_state_instantly = false;
        shared_ptr<Node> cur_node = MCTS::get_init_node();
        bool debug = true;

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
