import torch
import numpy as np
from time import sleep

from deoxys.franka_interface import FrankaInterface
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.transform_utils import quat2axisangle, mat2euler, mat2quat, quat_distance
from examples.osc_control import move_to_target_pose, deltas_move
from deoxys.experimental.motion_utils import reset_joints_to

logger = get_deoxys_example_logger()  # logger for debugging

robot_interface = FrankaInterface("/home/ripl/deoxys_control/deoxys/config/charmander.yml", use_visualizer=False, listen_cmds=True)  # hardcoded path to config file, probably should change
controller_type = "OSC_POSE"  # controls end effector in 6 dimensions, need to use serpeate controller for gripper
controller_cfg = get_default_controller_config(controller_type=controller_type)

# Golden resetting joints
reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]


reset_joints_to(robot_interface, reset_joint_positions)  # reset joints to home position
robot_interface.clear_cmd_buffer()


# deltas = [x, y, z, roll, pitch, yaw, gripper]
deltas = np.array([0.1, 0, 0, 0, 0, 0, 0])
for i in range(1):
    current_pose = np.concatenate((robot_interface.last_eef_quat_and_pos[1].flatten(), quat2axisangle(robot_interface.last_eef_quat_and_pos[0])))

    # deoxys_control
    # move_to_target_pose(
    #         robot_interface,
    #         controller_type,
    #         controller_cfg,
    #         target_delta_pose=deltas[:6],
    #         num_steps=5,
    #         num_additional_steps=0,
    #         interpolation_method="linear",
    #     )

    # directly command robot interface and sleep til within tolerance
    deltas_move(
        robot_interface,
        controller_type,
        controller_cfg,
        deltas,
    )

    # directly command robot interface
    # deltas[:3] = deltas[:3] / 0.05
    # robot_interface.control(
    #         controller_type=controller_type,
    #         action=deltas,
    #         controller_cfg=controller_cfg,
    #         verbose=False
    # )



    result_pos = np.concatenate((robot_interface.last_eef_quat_and_pos[1].flatten(), quat2axisangle(robot_interface.last_eef_quat_and_pos[0])))
    print("Result Delta:")
    print("before:", current_pose)
    print("after:", result_pos)
    cmds = robot_interface.tcp_command_buffer
    breakpoint()

robot_interface.close()