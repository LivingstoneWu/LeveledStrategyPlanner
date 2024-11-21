class EnvConstants:
    OBSERVATION_INDICES ={
        'time_left': [0, 1],
        'joint_positions': [1, 10],
        'joint_velocities': [10, 19],
        'end_effector_positions': [19, 28],
        # len 17
        'tool_object': {
            'type': 1,
            'size': 3,
            'cartesian_position': 3,
            'quaternion_orientation': 4,
            'linear_velocity': 3,
            'angular_velocity': 3
        },
        # len 11
        'goal_subshape': {
            'type': 1,
            'size': 3,
            'cartesian_position': 3,
            'orientation': 4
        },
        # len 11
        'fixed_block': {
            'type': 1,
            'size': 3,
            'cartesian_position': 3,
            'orientation': 4
        },
        # special case for reaching task
        'goal_positions': [28, 37]
    }

    TASK_PARAMS = {
        'pushing': {
            'num_objects': 1,
            'num_goals': 1,
            'num_blocks': 0
        },
        'picking': {
            'num_objects': 1,
            'num_goals': 1,
            'num_blocks': 0
        },
        'pick_and_place': {
            'num_objects': 1,
            'num_goals': 1,
            'num_blocks': 1
        },
        'stacking2': {
            'num_objects': 2,
            'num_goals': 2,
            'num_blocks': 0
        },
        'towers': {
            'num_objects': 5,
            'num_goals': 5,
            'num_blocks': 0
        }

    }