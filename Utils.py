from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task

# function to create vectorised environments
def make_env(env_id: str, rank: int, seed=0):
    def _init():
        env = CausalWorld(task=generate_task(task_generator_id=env_id), enable_visualization=False)
        env.seed(seed + rank)
        return env
    return _init