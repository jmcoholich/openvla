''' Use openVLA with camera feed, TODO: move robot with outputs'''

# Camera
import cv2
from PIL import Image
from openteach.utils.network import ZMQCameraSubscriber

# openVLA
from transformers import AutoModelForVision2Seq, AutoProcessor
from experiments.robot.robot_utils import get_vla_action, normalize_gripper_action

# deoxys_control
from deoxys.franka_interface import FrankaInterface
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
from deoxys.utils.config_utils import get_default_controller_config
from examples.osc_control import move_to_target_pose
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.transform_utils import quat2axisangle, mat2quat, euler2mat

# General
import torch
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import os
import json
from easydict import EasyDict


DEFAULT_CONTROLLER = EasyDict({
    'controller_type': 'OSC_POSE',
    'is_delta': True,
    'traj_interpolator_cfg': {
        'traj_interpolator_type': 'LINEAR_POSE',
        'time_fraction': 0.3
    },
    'Kp': {
        'translation': [250.0, 250.0, 250.0],
        'rotation': [250.0, 250.0, 250.0]
    },
    'action_scale': {
        'translation': 0.5,
        'rotation': 1.0
    },
    'residual_mass_vec': [0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5],
    'state_estimator_cfg': {
        'is_estimation': False,
        'state_estimator_type': 'EXPONENTIAL_SMOOTHING',
        'alpha_q': 0.9,
        'alpha_dq': 0.9,
        'alpha_eef': 1.0,
        'alpha_eef_vel': 1.0
    }
})

def main():
    # Initialize robot
    logger = get_deoxys_example_logger()  # logger for debugging

    robot_interface = FrankaInterface(
        os.path.join('/home/ripl/openteach/configs', 'deoxys.yml'), use_visualizer=False,
        control_freq=60,
        state_freq=200
    )  # copied from playback_demo.py

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

    easy_start = [ 0.04542551, 0.41048467, 0.0139709, -1.83497724, -0.14513091, 2.2443767, 1.06756333]
    reset_joints_to(robot_interface, easy_start)  # reset joints to home position


    # Load Processor & VLA
    model_path = "/home/ripl/openvla/runs/openvla-7b+franka_pick_coke+b8+lr-2e-05+lora-r32+dropout-0.0+new_recording+coke1"
    vla_path = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # should use all available GPUs
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)
    model.norm_stats = norm_stats
    # breakpoint()

    unnorm_key = "franka_pick_coke"
    task_label = "pick up the coke can"  # task to perform

    # Configure Camera Stream
    image_subscriber = ZMQCameraSubscriber(
            host = "143.215.128.151",
            port = "10007",  # 5 - top, 6 - side, 7 - front
            topic_type = 'RGB'
        )
    # np array for storing action values
    summary = None

    # Main loop
    try:
        # can I feed in training images and see how the robot moves
        while True:
            # Wait for a color frame
            frames = image_subscriber.recv_rgb_image()
            color_frame = frames[0]

            if color_frame is None:
                continue

            color_frame = color_frame[:, 140:500]  # center crop 360x360

            observation = {"full_image": color_frame}

            # pass through model
            action = get_vla_action(
                model,
                processor,
                "openvla",
                observation,
                task_label,
                unnorm_key,
                center_crop=True
            )
            # action[3:6] = quat2axisangle(mat2quat(euler2mat(action[3:6])))  # convert euler to axis-angle
            action = normalize_gripper_action(action, binarize=True)  # normalize gripper action
            print(f"predicted: {action}")

            # Move robot
            robot_interface.control(
                controller_type='OSC_POSE',
                action=action[:6],
                controller_cfg=DEFAULT_CONTROLLER,
            )
            robot_interface.gripper_control(-action[-1])

            # Display the image Press 'q' to exit
            cv2.imshow("Camera", color_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # graph action values on a line graph
            # if summary is None:
            #     summary = action
            # else:
            #     summary = np.vstack((summary, action))

    finally:
        cv2.destroyAllWindows()

        # Create line graph from summary
        # plt.plot(summary)
        # plt.xlabel('Frame')
        # plt.ylabel('Action Value')
        # plt.title('Action Values Over Time')
        # plt.legend(['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'])
        # plt.show()

if __name__ == "__main__":
    main()