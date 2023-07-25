from stable_baselines3 import PPO
from causal_world.envs.causalworld import CausalWorld
import causal_world.task_generators as tg
import causal_world.viewers.task_viewer as viewer

task_id = 'pushing'
model_name = 'ppo'

def demo():
    model = PPO.load('../Models/'+ model_name + '_' + task_id + '.zip')
    task = tg.generate_task(task_generator_id=task_id)
    world_params = dict()
    world_params['skip_frame'] = 1
    world_params['enable_visualization'] = True
    world_params['seed'] = 7

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    viewer.view_policy(task=task,
                       world_params=world_params,
                       policy_fn=policy_fn,
                       max_time_steps=40*960,
                       number_of_resets=40)

if __name__ == '__main__':
    demo()
    # model=PPO.load('../Models/ppo_pushing.zip')
    # print(model.policy)