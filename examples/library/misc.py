import argparse
import json
import ast
import os
import numpy as np
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Policy Iteration with output directory option')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory path (default: current directory)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed value for random number generator (default: 42)')
    parser.add_argument('--load-policy', type=str, default=None,
                        help='Path to a text file with a saved policy (default: None)')
    parser.add_argument('--load-env', type=str, default=None,
                        help='Path to a text file with a saved environment (default: None)')
    parser.add_argument('--radius', type=int, default=4,
                        help='Radius of the neighborhood of the goal state to optimize (default: 4)')
    args = parser.parse_args()
    return args


def save_world_to_file(env, file_path):
    with open(os.path.join(file_path, 'exp_world.txt'), 'w') as file:
        for row in env.world:
            file.write(''.join(row) + '\n')

def load_world_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = ''.join(file.readlines())
        lines = '\n' + lines  # Add '\n' at the beginning
        modified_string = lines.replace('\n', '\n    ')
        # world_txt = [line.strip() for line in lines]
    return modified_string


def load_policy_from_file(env, file_path):
    with open(file_path, 'r') as file:
        policy_dict_str = file.read()
        policy_dict = json.loads(policy_dict_str)
    pi = np.zeros(len(policy_dict))
    for key, val in policy_dict.items():
        tuple_key = ast.literal_eval(key)
        pi[env.state_dict[tuple_key]['state']] = val
    pi = np.array(pi)
    return pi.astype(np.int)

def save_policy_dict(env, path, policy):
    # Create a new dictionary to store the policy values
    policy_start_dict = {}

    # Iterate over the keys in env.state_dict
    for key in env.state_dict:
        # Retrieve the state information from env.state_dict
        state_info = env.state_dict[key]
        # Retrieve the corresponding policy value from pi using the state key
        policy_value = str(policy[state_info['state']])
        # Store the policy value in policy_start_dict with the state key
        str_key = str(key)
        policy_start_dict[str_key] = policy_value

    # Define the file path
    file_path = os.path.join(path, "policy_start_dict.txt")

    # Write the policy_start_dict to the file
    with open(file_path, 'w') as file:
        # Use json.dump() to write the dictionary to the file
        json.dump(policy_start_dict, file)


def find_goal(env):
    goal_keys = []
    for key, value in env.state_dict.items():
        # Check if "reward" is 100 and "done" is True
        if value.get('reward') == 100 and value.get('done') == True:
            goal_keys.append(key)
    return goal_keys

def find_goal_area(env, goal_coors, radius):
    subset_keys = []
    subset_indices = []
    for goal in goal_coors:
        for i in range(goal[0] - radius, goal[0] + radius + 1):
            for j in range(goal[1] - radius, goal[1] + radius + 1):
                if (i,j) in env.state_dict:
                    subset_keys.append((i, j))
                    subset_indices.append(env.state_dict[(i,j)]['state'])
    return subset_indices