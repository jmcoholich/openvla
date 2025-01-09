''' Use openVLA with camera feed, TODO: move robot with outputs'''

# Camera
import cv2
from PIL import Image
from openteach.utils.network import ZMQCameraSubscriber

# openVLA
from transformers import AutoModelForVision2Seq, AutoProcessor
from experiments.robot.robot_utils import get_vla_action, normalize_gripper_action
from prismatic.vla.action_tokenizer import ActionTokenizer

# deoxys
from deoxys.franka_interface import FrankaInterface
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.transform_utils import quat2axisangle, mat2quat, euler2mat
from deoxys.experimental.motion_utils import reset_joints_to

from openteach.utils.timer import FrequencyTimer

# jaca data
import pickle as pkl

# General
import torch
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import argparse
import math
import os
import json
import sys
import importlib
import tensorflow_datasets as tfds
from easydict import EasyDict


PI = np.pi
EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

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
    robot_interface = FrankaInterface(
        os.path.join('/home/ripl/openteach/configs', 'deoxys.yml'), use_visualizer=False,
        control_freq=1,
        state_freq=200
    )  # copied from playback_demo.py
    reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

    # reset_joints_to(robot_interface, reset_joint_positions)  # reset joints to home position

    prompt = "pick up the coke can"

    # Load demonstration data
    sys.path.append("/home/ripl/tensorflow_datasets")

    # module = importlib.import_module("franka_pick_coke")
    ds = tfds.load("franka_pick_coke", split='train')

    # Load Processor & VLA
    # model_path = "/home/ripl/openvla/runs/openvla-7b+franka_pick_coke+image_aug+5Hz_20_demo+RGB+Euler+-11gripper+224"
    # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForVision2Seq.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",  # should use all available GPUs
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True
    # )
    # # Create Action Tokenizer
    # action_tokenizer = ActionTokenizer(processor.tokenizer)

    # dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
    # with open(dataset_statistics_path, "r") as f:
    #     norm_stats = json.load(f)

    # model.norm_stats = norm_stats
    unnorm_key = "franka_pick_coke"

    # np array for storing action values
    summary_pred = None
    summary_true = None
    accuracy = 0

    # Main loop
    try:
        for i, episode in enumerate(ds.take(1)):
            for j, st in enumerate(episode['steps']):
                frame = st['observation']['image'].numpy()
                # frame = cv2.resize(frame, (224, 224))  # for some reason the image processor expects 224x224 not 360x360
                observation = {
                            "full_image": frame,
                            "state": None,
                        }

                recorded_action = st['action'].numpy()  # the action are deltas for (x, y, z, r, p, y, gripper)

                # get predicted action
                # action = get_vla_action(
                #     model,
                #     processor,
                #     "openvla",
                #     observation,
                #     prompt,
                #     unnorm_key,
                #     False
                # )
                # breakpoint()
                action = normalize_gripper_action(action, binarize=True)
                # action[3:6] = quat2axisangle(mat2quat(euler2mat(action[3:6])))
                recorded_action = normalize_gripper_action(recorded_action, binarize=True)
                # recorded_action[3:6] = quat2axisangle(mat2quat(euler2mat(recorded_action[3:6])))

                # Move robot
                # robot_interface.control(
                #         controller_type='OSC_POSE',
                #         action=action[:6],
                #         controller_cfg=DEFAULT_CONTROLLER,
                #     )
                # robot_interface.gripper_control(action[6])

                # # Predict Action (7-DoF; un-normalize)
                # print(f"Predicted: {action}")
                # print(f"Recorded:  {recorded_action}")
                # print(f"Match:   {action_tokenizer(action) == action_tokenizer(recorded_action)}")
                # accuracy = accuracy + (1 if action_tokenizer(action) == action_tokenizer(recorded_action) else 0)
                # print(f"Accuracy: {accuracy / (j + 1)}")



                # graph action values on a line graph
                if summary_pred is None:
                    summary_pred = action
                    summary_true = recorded_action
                else:
                    summary_pred = np.vstack((summary_pred, action))
                    summary_true = np.vstack((summary_true, recorded_action))


                # Display the image Press 'q' to exit
#                cv2.imshow("Camera", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#                if cv2.waitKey(1) & 0xFF == ord('q'):
#                    break
        print(j)
    finally:
        pass
#        cv2.destroyAllWindows()
        # Graph overlapping elements from summary_pred and summary_true
        # breakpoint()
#        titles = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
#        plt.figure(figsize=(12, 8))
#        for i in range(summary_pred.shape[1]):
#            plt.subplot(2, 4, i+1)
#            plt.plot(summary_pred[:, i], label=f"Pred")
#            plt.plot(summary_true[:, i], label=f"True")
#            plt.legend()
#            plt.xlabel("Step")
#            plt.ylabel("Action Value")
#            plt.title(f"{titles[i]}")
#            plt.ylim(-1.1, 1.1)
#            plt.legend(loc='lower right')
#        plt.tight_layout()
#        plt.show()


if __name__ == "__main__":
    main()
