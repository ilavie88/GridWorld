import numpy as np
from library.gridenv import small_env_fn, load_env_fn
import matplotlib.pyplot as plt
from library.helper import inf_norm,getMRP
from PIL import Image
import os
from library.misc import parse_arguments, load_policy_from_file, find_goal,\
    find_goal_area, save_policy_dict, save_world_to_file

from local_policy_itr import run_local_PI
from policy_itr import run_PI

def find_closest_state(env, a, b, radius):
    if (a, b) in env.state_dict:
        return (a, b)
    min_dist = 2*radius
    min_a, min_b = radius, radius
    for i in range(a - radius, a + radius + 1):
        for j in range(b - radius, b + radius + 1):
            if (i, j) in env.state_dict:
                dist = abs(a - i) + abs(b - j)
                if dist < min_dist:
                    min_a, min_b = i, j
                    min_dist = dist
    return min_a, min_b


def split_env_states(env, radius):
    subsets = []
    m = env.row // (2*radius)
    n = env.col // (2*radius)
    for i in range(m):
        for j in range(n):
            a = radius + i*radius
            b = radius + j*radius
            a, b = find_closest_state(env, a, b, radius)
            subset = find_goal_area(env, [(a, b)], radius)
            subsets.append(subset)
    return subsets


# Distributed Policy Iteration
def run_distributed_local_PI(env, output_path, seed=42, radius=4, init_policy=None):
    # path = os.path.join(output_path, "results")
    subsets = split_env_states(env, radius)

    # init_policy
    if not init_policy:
        init_policy = np.random.choice(env.action_values, size=env.state_count)

    policies = []
    for i, subset in enumerate(subsets):
        subset_path = os.path.join(output_path, f"subset_{i}")
        os.makedirs(subset_path, exist_ok=True)
        os.makedirs(os.path.join(subset_path, 'results'), exist_ok=True)
        run_local_PI(env, subset_path, subset, seed+1, init_policy)
        policy_path = os.path.join(subset_path, 'results', 'policy_start_dict.txt')
        policy = load_policy_from_file(env, policy_path)
        policies.append(policy)
    print('Reached here')
    for i, policy in enumerate(policies):
        for state in subsets[i]:
            init_policy[state] = policy[state]

    full_PI_path = os.path.join(output_path, 'full_PI')
    os.makedirs(full_PI_path, exist_ok=True)
    os.makedirs(os.path.join(full_PI_path, 'results'), exist_ok=True)

    save_policy_dict(env, full_PI_path, init_policy)
    run_PI(env, full_PI_path, os.path.join(full_PI_path, 'policy_start_dict.txt'))








if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed
    np.random.seed(seed)
    if args.load_env:
        env = load_env_fn(args.seed, args.load_env)
    else:  # Default use small_env_fn
        env = small_env_fn(args.seed)

    if args.load_policy:
        init_policy = load_policy_from_file(env, args.load_policy)
    else:
        init_policy = None

    radius = args.radius

    run_distributed_local_PI(env, args.output_dir, args.seed, radius, init_policy)
