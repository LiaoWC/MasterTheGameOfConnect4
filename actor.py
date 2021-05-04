from mcts_with_simulation import MCTS, NN3DConnect4
import os
import pathlib
import yaml
import argparse
import time
import torch
import json
import copy
from datetime import datetime


def get_filenames(dir_path):
    filenames = []
    for roots, dirs, files in os.walk(dir_path):
        for file in files:
            filenames.append(file)
    print(filenames)


# Save pt file:
# torch.save(model.state_dict(), f=latest_model_path)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', help='Path to config file.', required=True)
    args = parser.parse_args()

    # Read config from training_config.yaml
    config = yaml.safe_load(pathlib.Path(args.config).read_text())

    while True:
        #####################
        # Fetch newest model
        #####################
        # Find the latest.pt file
        latest_model_path = os.path.join(config['path']['models_dir'], 'latest.pt')
        if os.path.exists(latest_model_path):
            model = NN3DConnect4(features_3d_in_channels=4,
                                 features_scalar_in_channels=4,
                                 channels=config['model']['channels'],
                                 blocks=config['model']['blocks'])
            model.load_state_dict(torch.load(latest_model_path))
        else:
            # Fetch again after a while
            print('Find no latest pt model file...(wait for a while)')
            time.sleep(5)
            continue
        #######################
        # Self-play a game
        #######################
        cur_node = MCTS.get_init_node()
        start_properties = copy.deepcopy(cur_node.properties)
        start_board = copy.deepcopy(cur_node.board)
        start_hands = copy.deepcopy(cur_node.hands)
        moves = []  # 2D list
        while True:
            # Check if terminal
            if cur_node.hands >= 64:
                break
            # Self-play
            mcts = MCTS(root_node=cur_node,
                        eval_func='nn',
                        model=model,
                        device=config['self_play']['device'],
                        max_time_sec=config['self_play']['mcts_time_limit'],
                        max_simulation_cnt=config['self_play']['mcts_max_simulation_cnt'],
                        not_greedy=True if cur_node.hands <= config['self_play']['temperature_drop'] else False)
            move, sim_cnt, time_used = mcts.run(return_simulation_cnt=True, return_time_used=True)
            print(cur_node.hands, move, time_used)
            moves.append(move.tolist())
            cur_node = cur_node.get_node_after_playing(move=move)
        print()
        ####################
        # Save the trajectory
        ####################
        # Make dictionary to store trajectory information
        trajectory = {'start_properties': start_properties,
                      'start_board': start_board.tolist(),
                      'start_hands': start_hands,
                      'moves': moves}
        # Use current time to be part of the filename (Low prob to collision)
        output_filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.json'
        output_path = os.path.join(config['path']['not_trained_trajectories'], output_filename)
        with open(output_path, 'w+') as file:
            json.dump(trajectory, file)
