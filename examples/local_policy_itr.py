import numpy as np
from library.gridenv import small_env_fn, load_env_fn
import matplotlib.pyplot as plt
from library.helper import inf_norm,getMRP
from PIL import Image
import os
from library.misc import parse_arguments, load_policy_from_file, find_goal,\
    find_goal_area, save_policy_dict, save_world_to_file



script_path = os.environ.get("SCRIPT_PATH")





#Local Policy Iteration
def run_local_PI(env, output_path, subset=[1,2,3], seed=42, init_policy=None):
    path = os.path.join(output_path, "results")
    np.random.seed(seed)
    gamma = 0.9

    V = np.zeros((env.state_count,1))
    if init_policy.any():
        pi = init_policy
    else:
        pi = np.random.choice(env.action_values,size=env.state_count) #random policy
    pi_prev = np.random.choice(env.action_values,size=env.state_count)

    i=0
    v_values=[]

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
        image.save(os.path.join(path, f"pi_{i}.png"))  # ilavie - new codeline

        # Update value function and policy iteration counter
        v_values.append(inf_norm(V))
        i += 1

        report=f"Converged in {i} iterations\n"
        report+=f"Pi_*= {pi}\n"
        report+=f"V_*= {V.flatten()}\n"
        with open(os.path.join(path,"report.txt"), "w") as f:f.write(report)
        print(report)
        save_policy_dict(env, path, pi)
        save_world_to_file(env, path)

        plt.plot(v_values,lw=3,ls='--')
        plt.ylabel('$|V|_{\infty}$',fontsize=16)
        plt.xticks(range(len(v_values)),labels=["$\pi_{"+f"{e}"+"}$" for e in range(len(v_values))])
        plt.xlabel('Policy',fontsize=16)
        plt.tight_layout()

        # plt.savefig("./logs/policy_itr/pi_itr_v.png")
        plt.savefig(os.path.join(path, "pi_itr_v.png")) # ilavie - changed



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
    gamma = 0.9

    goal_coor = find_goal(env)
    subset = find_goal_area(env, goal_coor, radius)


    run_local_PI(env, args.output_dir, subset, seed, init_policy)
