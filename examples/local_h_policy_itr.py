import numpy as np
from library.gridenv import small_env_fn, load_env_fn
import matplotlib.pyplot as plt
from library.helper import inf_norm, getMRP
from PIL import Image
import os
from library.misc import parse_arguments, load_policy_from_file, save_policy_dict,\
    load_world_from_txt, save_world_to_file


# Added code to make sure file exists
# Construct the full file path using SCRIPT_PATH

def run_local_h_PI(env, args, subset=[1,2,3], eval_subset=[1,2,3], saved_policy_name='policy_state_dict.txt'):
    output_path = args.output_dir
    policy_path = args.load_policy
    save_only_last_img = args.save_only_last_img
    lookahead = args.lookahead
    gamma = args.gamma
    exact_eval = args.exact_evaluation


    path = os.path.join(output_path, "results")

    #Policy Iteration
    V = np.zeros((env.state_count, 1))
    if policy_path:
        pi = np.array(load_policy_from_file(env, policy_path))
        pi = pi.astype(np.int)
    else:
        pi = np.random.choice(env.action_values,size=env.state_count) #random policy
    pi_prev = np.random.choice(env.action_values,size=env.state_count)

    i=0
    v_values=[]

    if exact_eval:
        P_ss, R_s = getMRP(env, pi)
        A = np.eye(P_ss.shape[0], P_ss.shape[1]) - gamma * P_ss
        V = np.matmul(np.linalg.inv(A), R_s)
    policy_dict = {}

    while np.sum(np.abs(pi-pi_prev))>0: #until no policy change
        pi_prev = pi.copy()
        policy_dict[pi_prev.tobytes()] = 1
        for j in range(lookahead):
            P_ss, R_s = getMRP(env,pi)
            if exact_eval:
                A = np.eye(P_ss.shape[0], P_ss.shape[1]) - gamma * P_ss
                V = np.matmul(np.linalg.inv(A), R_s)
            else:
                V_eval = R_s[eval_subset] + gamma * np.matmul(P_ss[eval_subset], V)
                V[eval_subset] = V_eval
                # V = eval_stationary_point(V.copy(), R_s, P_ss, gamma, eval_subset)

            # Extract a subset of states to optimize
            subset_V = V[subset]
            subset_P_sas = env.P_sas[subset][:, :, subset]
            subset_R_sa = env.R_sa[subset]

            # Policy lookahead optimization
            subset_pi = np.argmax(subset_R_sa + gamma * np.squeeze(np.matmul(subset_P_sas, subset_V)), axis=1)
            pi[subset] = subset_pi

        if not save_only_last_img:
            image = Image.fromarray(env.getScreenshot(pi))
            image.save(os.path.join(path, f"pi_{i}.png")) # ilavie - new codeline
        v_values.append(inf_norm(V))
        i+=1
        if pi.tobytes() in policy_dict:
            break

    if save_only_last_img:
        image = Image.fromarray(env.getScreenshot(pi))
        image.save(os.path.join(path, f"pi_{i}.png"))
    report = f"Converged in {i} iterations\n"
    report += f"Pi_*= {pi}\n"
    report += f"V_*= {V.flatten()}\n"
    with open(os.path.join(path, "report.txt"), "w") as f:f.write(report)
    print(report)
    save_policy_dict(env, path, pi)
    save_world_to_file(env, path)

    plt.plot(v_values,lw=3,ls='--')
    plt.ylabel('$|V|_{\infty}$',fontsize=16)
    plt.xticks(range(len(v_values)),labels=["$\pi_{"+f"{e}"+"}$" for e in range(len(v_values))])
    plt.xlabel('Policy',fontsize=16)
    plt.tight_layout()

    # plt.savefig("./logs/policy_itr/pi_itr_v.png")
    plt.savefig(os.path.join(path,"pi_itr_v.png")) # ilavie - changed
    number_of_iterations = i
    return pi, V, number_of_iterations



if __name__ == "__main__":
    args = parse_arguments()
    np.random.seed(args.seed)
    if args.load_env:
        env = load_env_fn(args.seed, args.load_env)
    else: # Default use small_env_fn
        env = small_env_fn(args.seed)

    run_h_PI(env, args)
