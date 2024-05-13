import argparse
import json
import ast
import os
import numpy as np
import random

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
    parser.add_argument('--save-generated-maze', type=str, default=None,
                        help='Path to save the generated maze (default: None, save in output directory)')
    parser.add_argument('--save-only-last-img', action='store_true',
                        help='Whether to save only the last image of output PI')
    parser.add_argument('--generated-maze-size', type=int, default=20,
                        help='Size of the generated maze (default: 20)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards (default: 0.99)')
    parser.add_argument('--evaluation-radius', type=int, default=None,
                        help='Evaluation radius for the policy (default: None)')
    parser.add_argument('--lookahead', type=int, default=1,
                        help='Lookahead parameter for policy evaluation (default: 1)')
    parser.add_argument('--exact-evaluation', action='store_true',
                        help='Whether to perform exact evaluation (default: True)')
    args = parser.parse_args()
    return args

def eval_stationary_point(V0, R, P, gamma, eval_subset, epsilon=1):
    V_prev = V0.copy()
    V = V0.copy()
    V_eval = R[eval_subset] + gamma * np.matmul(P[eval_subset], V)
    V[eval_subset] = V_eval
    while np.linalg.norm(V - V_prev) > epsilon:
        V_prev = V.copy()
        V_eval = R[eval_subset] + gamma * np.matmul(P[eval_subset], V)
        V[eval_subset] = V_eval
    return V

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
        if tuple_key in env.state_dict:
            pi[env.state_dict[tuple_key]['state']] = val
    pi = np.array(pi)
    return pi.astype(np.int)

def save_policy_dict(env, path, policy, policy_name="policy_state_dict"):
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
    file_path = os.path.join(path, policy_name)

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

def generate_maze(maze_size, num_walls, num_holes, file_path):
    # Initialize maze grid with a frame of 'w'
    maze = [[' ' for _ in range(maze_size)] for _ in range(maze_size)]
    for i in range(maze_size):
        maze[i][0] = 'w'
        maze[i][maze_size - 1] = 'w'
        maze[0][i] = 'w'
        maze[maze_size - 1][i] = 'w'


    # Place 'a' (agent) randomly in the maze
    agent_pos = (random.randint(1, maze_size - 2), random.randint(1, maze_size - 2))
    maze[agent_pos[0]][agent_pos[1]] = 'a'

    # Place 'g' (goal) randomly in the maze
    goal_pos = (random.randint(1, maze_size - 2), random.randint(1, maze_size - 2))
    while goal_pos == agent_pos:
        goal_pos = (random.randint(1, maze_size - 2), random.randint(1, maze_size - 2))
    maze[goal_pos[0]][goal_pos[1]] = 'g'


    # Fill maze with walls
    for i in range(num_walls):
        found_wall_pos = False
        while not found_wall_pos:
            wall_pos = (random.randint(1, maze_size - 2), random.randint(1, maze_size - 2))
            if maze[wall_pos[0]][wall_pos[1]] == ' ':
                maze[wall_pos[0]][wall_pos[1]] = 'w'
                found_wall_pos = True

    # Fill maze with holes
    for i in range(num_holes):
        found_hole_pos = False
        while not found_hole_pos:
            hole_pos = (random.randint(1, maze_size - 2), random.randint(1, maze_size - 2))
            if maze[hole_pos[0]][hole_pos[1]] == ' ':
                maze[hole_pos[0]][hole_pos[1]] = 'o'
                found_hole_pos = True


    # Save maze to a text file
    with open(file_path, 'w') as file:
        for row in maze:
            file.write(''.join(row) + '\n')


if __name__ == "__main__":
    args = parse_arguments()
    maze_size = args.generated_maze_size
    num_walls = maze_size*maze_size // 5
    # num_holes = maze_size*maze_size // 50
    num_holes = 0
    generate_maze(maze_size, num_walls, num_holes, args.save_generated_maze)