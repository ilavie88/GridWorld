import numpy as np
from library.gridenv import small_env_fn, train_env_fn
import matplotlib.pyplot as plt
from library.helper import inf_norm,getMRP
from PIL import Image
import os

from examples.local_policy_itr import local_PI, find_goal, find_goal_area



script_path = os.environ.get("SCRIPT_PATH")

def dfs(env, start_state, h):
    stack = [start_state]
    visited = set()
    dist = 0

    while stack:
        curr_state = stack.pop()
        if curr_state in visited:
            continue

        visited.add((curr_state, i))

        # Get neighboring states
        neighbors = get_neighbors(env, curr_state)

        # Add unvisited neighbors to the stack
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)

def get_neighbors(env, state):
    i, j = state
    neighbors = []

    # Define possible moves (up, down, left, right)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for di, dj in moves:
        new_i, new_j = i + di, j + dj
        if (new_i, new_j) in env.good_states_dict:
            neighbors.append((new_i, new_j))

    return neighbors

# Example usage:
# Assuming env is your environment object and start_state is the starting state
# dfs(env, start_state)


def is_empty_state(state_discrition):
    if state_discrition['reward'] == -1 and state_discrition['done'] == false:
        return True
    return False

def get_env_bounds(env):
    min_col = np.inf
    max_col = - np.inf
    min_row = np.inf
    max_row = - np.inf
    for (key, val) in env.state_dict.items():
        if key[0] < min_col:
            min_col = key[0]
        if key[0] > max_col:
            max_col = key[0]
        if key[1] < min_row:
            min_row = key[1]
        if key[1] > max_row:
            max_row = key[1]
    return min_col, max_col, min_row, max_row


def init_anchor(env):
    for (kev, value) in env.state_dict.items():
        if is_empty_state(value):
            return key
    print("Couldn't find an empty slot on the env.")
    return (-1, -1)

def go_right(env, curr_anchor, distance):
    anchor_i = curr_anchor[0]
    anchor_j = curr_anchor[1]
    traj_stack = []
    while (anchor_i, anchor_j) in env.good_state_dict and distance > 0:
        traj_stack.append(anchor_i, anchor_j)
        anchor_i += 1
        distance -= 1
    return (anchor_i, anchor_j), distance, traj_stack




def find_distance_anchor(env, distance, curr_anchor, first_action):
    """
    This function finds the most distance available anchor from curr_anchor starting in the
     action first_action
    """
    anchor_i = curr_enchor[0]
    anchor_j = curr_enchor[1]
    path_to_anchor = []
    if first_action == 0:
        while distance > 0:
            if (anchor_i+1, anchor_j) in env.good_state_dict:
                anchor_i += 1
                path_to_anchor.append((anchor_i, anchor_j))




# Greedy anchor search
def greedy_anchors_search(env, distance):
    min_col, max_col, min_row, max_row = get_env_bounds(env)

    pi = np.random.choice(env.action_values, size=env.state_count)  # random policy
    anchors = []
    anchors.append(init_anchor(env))
    for i, anchor in enumerate(anchors):
        for action_type in env.action_values
            if action_type == 0: # go to the right
                if anchor[1] + distance < max_col:




#Local Globaal Policy Iteration
def local_global_PI(env, distance, seed=42):
    np.random.seed(seed)
    gamma = 0.9
    anchors = greedy_anchors_search(env, distance)

    V = np.zeros((env.state_count,1))
    pi = np.random.choice(env.action_values,size=env.state_count) #random policy
    pi_prev = np.random.choice(env.action_values,size=env.state_count)

    i=0
    v_values=[]

    # ChatGPT code
    while np.sum(np.abs(pi - pi_prev)) > 0:  # until no policy change
        pi_prev = pi.copy()
        P_ss, R_s = getMRP(env, pi)
        A = np.eye(P_ss.shape[0], P_ss.shape[1]) - gamma*P_ss
        V = np.matmul(np.linalg.inv(A),R_s)

        # Extract a subset of states
        subset_V = V[subset]
        subset_P_sas = env.P_sas[subset][:, :, subset]
        subset_R_sa = env.R_sa[subset]

        # Policy optimization on the subset of states
        subset_pi = np.argmax(subset_R_sa + gamma * np.squeeze(np.matmul(subset_P_sas, subset_V)), axis=1)
        pi[subset] = subset_pi

        # Save policy iteration screenshot
        image = Image.fromarray(env.getScreenshot(pi))
        image.save(os.path.join(script_path, 'logs', 'local_policy_itr', f"pi_{i}.png"))  # ilavie - new codeline

        # Update value function and policy iteration counter
        v_values.append(inf_norm(V))
        i += 1

        report=f"Converged in {i} iterations\n"
        report+=f"Pi_*= {pi}\n"
        report+=f"V_*= {V.flatten()}\n"
        with open(os.path.join(script_path, "logs", "local_policy_itr","report.txt"), "w") as f:f.write(report)
        print(report)

        plt.plot(v_values,lw=3,ls='--')
        plt.ylabel('$|V|_{\infty}$',fontsize=16)
        plt.xticks(range(len(v_values)),labels=["$\pi_{"+f"{e}"+"}$" for e in range(len(v_values))])
        plt.xlabel('Policy',fontsize=16)
        plt.tight_layout()

        # plt.savefig("./logs/policy_itr/pi_itr_v.png")
        plt.savefig(os.path.join(script_path, "logs", "local_policy_itr","pi_itr_v.png")) # ilavie - changed


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    distance = 4
    gamma = 0.9

    env = train_env_fn(seed)

    local_global_PI(env, distance)
