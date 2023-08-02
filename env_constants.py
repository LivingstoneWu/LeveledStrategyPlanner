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
