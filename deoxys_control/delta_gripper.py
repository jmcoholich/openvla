''' Method to control the gripper of the robot with deltas from openVLA. '''

import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
import deoxys.proto.franka_interface.franka_robot_state_pb2 as franka_robot_state_pb2

import numpy as np


def reset_gripper(robot_interface, logger):
    """Reset the gripper to open state

    Args:
        robot_interface (FrankaInterface): The interface to the robot
        logger (Logger): The logger object
    """

    gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()

    move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
    move_msg.width = 0.08
    move_msg.speed = 0.1
    gripper_control_msg.control_msg.Pack(move_msg)

    logger.debug("Gripper opening")

    robot_interface._gripper_publisher.send(gripper_control_msg.SerializeToString())
    robot_interface.last_gripper_action = 1

def delta_gripper(robot_interface, logger, action: float, delta: float = 0.005):
    """Control the gripper

    Args:
        action (float): The control command for Franka gripper. Currently assuming scalar control commands.
    """

    gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()

    # action = 1 : Opening
    # action = 0 : Closing

    # TODO (Yifeng): Test if sending grasping or gripper directly
    # will stop executing the previous command
    if np.isclose(action, 1, .1):  # start opening
        move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
        move_msg.width = min(robot_interface.last_gripper_q + delta, 0.08)
        move_msg.speed = 0.1
        gripper_control_msg.control_msg.Pack(move_msg)

        logger.debug("Gripper opening")

        robot_interface._gripper_publisher.send(gripper_control_msg.SerializeToString())
    elif np.isclose(action, 0, .1):  # start closing
        grasp_msg = franka_controller_pb2.FrankaGripperMoveMessage()
        grasp_msg.width = max(robot_interface.last_gripper_q - delta, 0)
        grasp_msg.speed = 0.1
        # grasp_msg.force = 30.0
        # grasp_msg.epsilon_inner = 0.08
        # grasp_msg.epsilon_outer = 0.08

        gripper_control_msg.control_msg.Pack(grasp_msg)

        logger.debug("Gripper closing")

        robot_interface._gripper_publisher.send(gripper_control_msg.SerializeToString())
    robot_interface.last_gripper_action = action