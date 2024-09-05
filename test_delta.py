import torch
import numpy as np
from time import sleep

from deoxys.franka_interface import FrankaInterface
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
from deoxys.utils.config_utils import get_default_controller_config
from deoxys_control.osc_control import move_to_target_pose, reset_joints_to  # borrowing this to move based on deltas
from deoxys_control.delta_gripper import delta_gripper, reset_gripper
from deoxys.utils.log_utils import get_deoxys_example_logger


logger = get_deoxys_example_logger()  # logger for debugging

robot_interface = FrankaInterface("/home/ripl/deoxys_control/deoxys/config/charmander.yml", use_visualizer=False)  # hardcoded path to config file, probably should change
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
reset_gripper(robot_interface, logger)  # reset gripper to open

breakpoint()
# deltas = [x, y, z, roll, pitch, yaw, gripper]
deltas = [.01, 0, 0, 0, 0, 0, 0]
for i in range(10):

    move_to_target_pose(
            robot_interface,
            controller_type,
            controller_cfg,
            target_delta_pose=deltas[:6],
            num_steps=40,
            num_additional_steps=0,
            interpolation_method="linear",
        )
    delta_gripper(robot_interface, logger, deltas[6])
    sleep(1)

robot_interface.close()