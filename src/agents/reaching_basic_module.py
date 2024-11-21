from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.task_generators import ReachingTaskGenerator
import numpy as np
from causal_world.utils.rotation_utils import cart2cyl, cyl2cart
from causal_world.envs.robot.trifinger import TriFingerRobot


# redefine a task of reaching nearby positions
class ReachNearbyGenerator(ReachingTaskGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_new_goal_pos(self, starting_positions_cart) -> dict:
        original_positions = starting_positions_cart

        new_goal = {}

        while True:
            # sampling by the fitted gaussian distribution, original mean and std are 0.088 and 0.042
            sampled_positions = original_positions + np.random.normal(0.088, 0.042, size=original_positions.shape)
            new_goal['goal_60'] = {'cylindrical_position': cart2cyl(sampled_positions[0:3])}
            new_goal['goal_120'] = {'cylindrical_position': cart2cyl(sampled_positions[3:6])}
            new_goal['goal_300'] = {'cylindrical_position': cart2cyl(sampled_positions[6:9])}
            if not super(ReachingTaskGenerator, self).is_intervention_in_bounds(new_goal):
                continue

        return new_goal

    # samples starting cartesian positions from the intervention space, then turn them into joint positions
    def sample_starting_positions(self):
        # Previous code: we would not return dict here anymore, but return the array of joint positions

        # intervention_dict = dict()
        # joint_positions_bounds = self.get_intervention_space_a_b()['joint_positions']
        # intervention_dict['joint_positions'] = np.random.uniform(joint_positions_bounds[0], joint_positions_bounds[1])
        # return intervention_dict

        tip_position_bounds = self.get_intervention_space_a_b()['goal_60']['cylindrical_position']
        positions = np.array([])
        for i in range(3):
            positions = np.append(positions,
                                  cyl2cart(np.random.uniform(tip_position_bounds[0], tip_position_bounds[1])))
        return self._robot.get_joint_positions_from_tip_positions(positions), positions


def init_env(dummy_task: ReachNearbyGenerator) -> CausalWorld:
    task = ReachNearbyGenerator(joint_positions=dummy_task.sample_starting_positions())
    env = CausalWorld(task=task, enable_visualization=False)
    env.reset()
    env.do_intervention(env.get_task().sample_new_goal())
    return env


if __name__ == '__main__':
    dummy_task = ReachNearbyGenerator()
    dummy_env = CausalWorld(task=dummy_task, enable_visualization=False)
    # observation = dummy_env.reset()
    # print(observation[19:28])
    # print(cart2cyl(observation[19:22]))
    # print(dummy_task._task_robot_observation_keys)
    # print(dummy_task._current_full_observations_dict)
    # print(dummy_task.get_intervention_space_a_b())

    start_joint_positions, start_cart_positions = dummy_task.sample_starting_positions()
    task = ReachNearbyGenerator(joint_positions=start_joint_positions)
    env = CausalWorld(task=task, enable_visualization=False)
    observation = env.reset()
    print(observation[19:22])
    print(start_cart_positions)
    print(start_joint_positions)
    print(env.get_task().get_task_params())
    joint_intervention = {'joint_positions': start_joint_positions}
    success_signal, new_observation = env.do_intervention(joint_intervention)
    print(new_observation[19:28])
    # print(task.sample_new_goal_pos(observation[19:28]))
    # env.do_intervention(task.sample_new_goal_pos(observation[19:28]))
    # print(observation[19:28])
    # print(env.get_task().get_desired_goal())
