from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import numpy as np
import matplotlib.pyplot as plt

generate_task(task_generator_id='reaching')
env = CausalWorld(task=generate_task(task_generator_id='reaching'), enable_visualization=False, action_mode='joint_torques')
observation= env.reset()
distances_list=[]

def distance(obs1, obs2):
    return np.linalg.norm(obs1-obs2)

for i in range(10000):
    obs, reward, done, info=env.step(env.action_space.sample())
    for j in range(3):
        distances_list.append(distance(observation[19:22+3*j], obs[19:22+3*j]))
    observation=obs

print("Mean distance: ", np.mean(distances_list))
print("Std distance: ", np.std(distances_list))

plt.hist(distances_list, bins=200, density=True)
plt.title("Distance the end-effector moved in consecutive\n time steps with random actions")
plt.xlabel("Distance")
plt.ylabel("Frequency")
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/(sig*np.sqrt(2*np.pi))
x = np.linspace(0, 0.25, 1000)
plt.plot(x, gaussian(x, np.mean(distances_list), np.std(distances_list)), color='red')
plt.legend(['Gaussian fit', 'Histogram'])
plt.show()
plt.savefig('./distances_dict.png', dpi=400)