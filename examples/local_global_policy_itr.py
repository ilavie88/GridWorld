import numpy as np
from library.gridenv import small_env_fn, train_env_fn
import matplotlib.pyplot as plt
from library.helper import inf_norm,getMRP
from PIL import Image
import os



script_path = os.environ.get("SCRIPT_PATH")





#Local Policy Iteration
def local_PI(env, subset=[1,2,3], seed=42):
    np.random.seed(seed)
    # env = small_env_fn(seed)
    gamma = 0.9
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


    # while np.sum(np.abs(pi-pi_prev))>0: #until no policy change
    #         pi_prev=pi.copy()
    #         P_ss,R_s=getMRP(env,pi)
    #         V=R_s+gamma*np.matmul(P_ss,V)
    #         pi=np.argmax(env.R_sa+gamma*np.squeeze(np.matmul(env.P_sas,V)),axis=1)
    #         image=Image.fromarray(env.getScreenshot(pi))
    #         # image.save(f"./logs/policy_itr/pi_{i}.png") # Changed due to some problem in saving the img
    #         image.save(os.path.join(script_path, 'logs', 'policy_itr', f"pi_{i}.png")) # ilavie - new codeline
    #         v_values.append(inf_norm(V))
    #         i+=1

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

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    radius = 2
    gamma = 0.9

    env = small_env_fn(seed)
    goal_coor = find_goal(env)
    subset = find_goal_area(env, goal_coor, radius)


    local_PI(env, subset, seed)
