import numpy as np
from library.gridenv import small_env_fn
import matplotlib.pyplot as plt
from library.helper import inf_norm,getMRP
from PIL import Image
import os

# Added code to make sure file exists
# Construct the full file path using SCRIPT_PATH
script_path = os.environ.get("SCRIPT_PATH")



np.random.seed(42)
env=small_env_fn(42)
gamma=0.9

#Policy Iteration
V=np.zeros((env.state_count,1))
pi=np.random.choice(env.action_values,size=env.state_count) #random policy
pi_prev=np.random.choice(env.action_values,size=env.state_count)

i=0
v_values=[]

while np.sum(np.abs(pi-pi_prev))>0: #until no policy change
    pi_prev=pi.copy()
    P_ss,R_s=getMRP(env,pi)
    V=R_s+gamma*np.matmul(P_ss,V)
    pi=np.argmax(env.R_sa+gamma*np.squeeze(np.matmul(env.P_sas,V)),axis=1)
    image=Image.fromarray(env.getScreenshot(pi))
    # image.save(f"./logs/policy_itr/pi_{i}.png") # Changed due to some problem in saving the img
    image.save(os.path.join(script_path, 'logs', 'policy_itr', f"pi_{i}.png")) # ilavie - new codeline
    v_values.append(inf_norm(V))
    i+=1

report=f"Converged in {i} iterations\n"
report+=f"Pi_*= {pi}\n"
report+=f"V_*= {V.flatten()}\n"
with open(os.path.join(script_path, "logs", "policy_itr","report.txt"), "w") as f:f.write(report)
print(report)

plt.plot(v_values,lw=3,ls='--')
plt.ylabel('$|V|_{\infty}$',fontsize=16)
plt.xticks(range(len(v_values)),labels=["$\pi_{"+f"{e}"+"}$" for e in range(len(v_values))])
plt.xlabel('Policy',fontsize=16)
plt.tight_layout()

# plt.savefig("./logs/policy_itr/pi_itr_v.png")
plt.savefig(os.path.join(script_path, "logs", "policy_itr","pi_itr_v.png")) # ilavie - changed
