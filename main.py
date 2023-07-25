from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import numpy as np

task = generate_task(task_generator_id='reaching')
env = CausalWorld(task=task, enable_visualization=True)
obs, reward, done, info = env.step(env.action_space.sample())
print(obs)
print("------------------")
print(reward)
print("------------------")
print(done)
print("------------------")
print(info)


# Sampling new goal
from stable_baselines3 import PPO
from causal_world.envs.causalworld import CausalWorld
import causal_world.task_generators as tg
from stable_baselines3.common.vec_env import SubprocVecEnv

task = tg.PushingTaskGenerator(variables_space='space_a')
env = CausalWorld(task=task, enable_visualization=True, seed=7, skip_frame=2)
env.reset()
success_signal, obs= env.do_intervention(env.sample_new_goal())
print(env.get_task().get_desired_goal())
for i in range(5000):
    obs, reward, done, info = env.step(env.action_space.sample())
