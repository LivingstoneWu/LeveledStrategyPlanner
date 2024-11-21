from stable_baselines3 import PPO
from causal_world.envs.causalworld import CausalWorld
import causal_world.task_generators as tg
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

# task = tg.PushingTaskGenerator(variables_space='space_a')
# env = CausalWorld(task=task, enable_visualization=True, seed=7, skip_frame=2)
# env.reset()
# success_signal, obs= env.do_intervention(env.sample_new_goal())
# print(env.get_task().get_desired_goal())
# for i in range(5000):
#     obs, reward, done, info = env.step(env.action_space.sample())


# create vectorised environments
def make_env(env_id: str, rank: int, seed=1428, variables_space='space_a', skip_frame=10, action_mode='joint_positions'):
    def _init():
        env = CausalWorld(task=tg.generate_task(env_id, variables_space=variables_space, dense_reward_weights=np.array([0,0,0])), enable_visualization=False, skip_frame=skip_frame,
                          action_mode=action_mode, max_episode_length=1000)
        env.seed(seed + rank)
        env.do_intervention(env.sample_new_goal())
        return env
    return _init

# train baseline model for reaching task with vectorised environments


if __name__=='__main__':
    env_id = 'pushing'
    num_cpu=4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', learning_rate=0.00025, policy_kwargs=dict(net_arch=[128,128]))
    model.learn(total_timesteps=1000000)
    model.save("ppo_pushing")
