import os
import numpy as np
import matplotlib.pyplot as plt

from library.gridenv import load_env_fn
from library.helper import getMRP

from library.misc import parse_arguments, save_policy_dict, load_policy_from_file, find_goal, find_goal_area
from examples.local_policy_itr import run_local_PI
from examples.policy_itr import run_PI

PATH_TO_MAZES = "/home/ilavie/local_PI/experiments/local_finetuning_exp/20dist/mazes"
EXP_PATH = "/home/ilavie/local_PI/experiments/local_finetuning_exp/20dist/"
OPTIMIZATION_RADIUS_RANGE = 20
HPI=True

def save_statistics(Values_vec, neighborhood_radiuses, iteration_vec, dir_path):

    np.savetxt(os.path.join(dir_path, 'iterations_and_radius.txt'), np.column_stack((iteration_vec, neighborhood_radiuses)))
    np.savetxt(os.path.join(dir_path, 's0_Value_function_and_radius.txt'), np.column_stack((Values_vec, neighborhood_radiuses)))

    # Clear the current plot
    plt.clf()  # or plt.cla()

    # Generate plot
    plt.plot(neighborhood_radiuses, Values_vec)
    plt.xlabel('Neighborhood radius')
    plt.ylabel('Normalized $V[s0]$')
    plt.title('Normalized $V[s0]$ as a Function of Radius')
    plt.grid(True)

    # Save plot as image file
    plt.savefig(os.path.join(dir_path, 'value_function_plot.png'))

    # Show plot (optional)
    plt.show()

def get_finetuning_conergence_statics(args, mazes_path, optimization_radius_range):
    # Scripts initialization:
    gamma = args.gamma

    # Iterate over all mazes in the directory
    for k, maze_dir in enumerate(os.listdir(mazes_path)):
        # Initialization
        V_s0 = np.zeros(optimization_radius_range + 1)
        iterations = np.zeros(optimization_radius_range + 1)


        # Join the directory path with the item to get the full path
        item_path = os.path.join(mazes_path, maze_dir)
        os.makedirs(os.path.join(item_path, 'results'), exist_ok=True)

        # Define the environment for this iteration
        env_to_load = os.path.join(item_path, 'mod_world_map.txt')
        env = load_env_fn(args.seed, env_to_load)
        args.load_env = env_to_load
        init_state = env.state_dict[tuple(env.agent.initial_position)]['state']

        # Solve optimal policy of modefied world for reference
        reference_path = os.path.join(mazes_path, maze_dir, 'refernce')
        os.makedirs(reference_path, exist_ok=True)
        os.makedirs(os.path.join(reference_path, 'results'), exist_ok=True)

        args.output_dir = reference_path
        args.load_env = env_to_load

        pi_star, v_star = run_PI(env, args)
        V_ref = v_star[init_state]

        # Find goal state for corruption
        goal_coor = find_goal(env)

        # Load optimal policy and create corrupted policy
        policy_to_load = os.path.join(item_path,'opt_policy_dict.txt')
        pi = load_policy_from_file(env, policy_to_load)
        # policy_star_to_load = os.path.join(item_path, 'mod_world_opt_policy.txt')
        # pi_star = load_policy_from_file(env, policy_star_to_load)


        # # Calculate optimal value function with optimal policy
        # P_ss, R_s = getMRP(env, pi_star)
        # A = np.eye(P_ss.shape[0], P_ss.shape[1]) - gamma * P_ss
        # V_star = np.matmul(np.linalg.inv(A), R_s)
        # V_ref = V_star[init_state]


        # Calculate initial value function with optimal policy
        P_ss, R_s = getMRP(env, pi)
        A = np.eye(P_ss.shape[0], P_ss.shape[1]) - gamma * P_ss
        V_corr = np.matmul(np.linalg.inv(A), R_s)
        V_s0[0] = V_corr[init_state] / V_ref

        # Iterate over all mazes
        for radius in range(1, optimization_radius_range + 1):
            # Create output dirs
            output_dir_path = os.path.join(item_path, f"optimization_radius{radius}")
            os.makedirs(output_dir_path, exist_ok=True)
            os.makedirs(os.path.join(output_dir_path, "results"), exist_ok=True)

            # Change required args for run
            args.radius = radius
            args.load_policy = env_to_load
            args.output_dir = output_dir_path
            args.load_policy = policy_to_load
            args.evaluation_radius = radius + 4

            # Define optimization subsets
            subset = find_goal_area(env, goal_coor, radius)
            eval_subset = find_goal_area(env, goal_coor, args.evaluation_radius)

            pi_radius, V_radius, iteration = run_local_PI(env, args, subset, eval_subset)

            # Evaluate final value function
            P_ss, R_s = getMRP(env, pi_radius)
            A = np.eye(P_ss.shape[0], P_ss.shape[1]) - gamma * P_ss
            V_radius = np.matmul(np.linalg.inv(A), R_s)

            V_s0[radius] += V_radius[init_state] / V_ref
            iterations[radius] += iteration
        # V_goal /= V_star[env.state_dict[tuple(goal_coor[0])]['state']]
        save_statistics(V_s0, range(optimization_radius_range + 1), iterations, item_path)



if __name__ == "__main__":
    args = parse_arguments()
    args.output_dir = EXP_PATH
    get_finetuning_conergence_statics(args, PATH_TO_MAZES, OPTIMIZATION_RADIUS_RANGE)

