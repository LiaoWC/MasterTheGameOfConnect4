import connect4_3d_ai_engine as engine

node = engine.get_init_node()
for i in range(10):
    mcts = engine.MCTS(root=node, max_simulation_cnt=99999999, max_time_sec=5, print_simulation_cnt=True)
    move = mcts.run(playout_use_block_move=False,
                    reward_add_score_diff=False,
                    first_hand_center=True,
                    dominate_pruning=True,
                    prior_value=True)
    node = node.get_node_after_playing(move)





