''' Use openVLA with camera feed, TODO: move robot with outputs'''

# Camera
import cv2
from PIL import Image
from openteach.utils.network import ZMQCameraSubscriber

# openVLA
from transformers import AutoModelForVision2Seq, AutoProcessor
from experiments.robot.robot_utils import get_vla_action

# deoxys_control
from deoxys.franka_interface import FrankaInterface
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
from deoxys.utils.config_utils import get_default_controller_config
from deoxys_control.osc_control import move_to_target_pose, reset_joints_to  # borrowing this to move based on deltas
from deoxys_control.delta_gripper import delta_gripper, reset_gripper
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.transform_utils import quat2axisangle

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

parser = argparse.ArgumentParser()
parser.add_argument("demo", type=str, help="The name of the demonstration to visualize")

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


def mat2euler(rmat, axes="sxyz"):  # from /home/ripl/deoxys_control/deoxys/deoxys/utils/transform_utils.py
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return np.array((ax, ay, az), dtype=np.float32)

def main(args):
    # load data
    # Load example data and print data structure


    # Load demonstration data
    filename = f"/home/ripl/openteach/extracted_data/demonstration_{args.demo}/demo_{args.demo}.pkl"
    with open(filename, 'rb') as dbfile:
        db = pkl.load(dbfile)

    images = db["rgb_imgs"][2]
    prompt = "pick up the coke can"
    breakpoint()

    # Load Processor & VLA
    model_path = "/home/ripl/openvla/runs/openvla-7b+franka_pick_coke+b8+lr-2e-05+lora-r32+dropout-0.0--image_aug"
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

    # make slight alterations to norm stats for testing purposes
    for key in norm_stats['franka_pick_coke']['action'].keys():
        if key == 'mask' or key == 'mean':
            continue
        norm_stats['franka_pick_coke']['action'][key] = [i * 10 for i in norm_stats['franka_pick_coke']['action'][key]]
    breakpoint()
    model.norm_stats = norm_stats
    unnorm_key = "franka_pick_coke"

    # np array for storing action values
    summary = None

    # Main loop
    try:
        for i, frame in enumerate(images):
            # Convert the frame to PIL Image
            # video_frame = cv2.resize(frame, (224, 224))
            image: Image.Image = Image.fromarray(frame)

            observation = {
                        "full_image": frame,
                        "state": None,
                    }
            # breakpoint()
            # Convert demonstration data to 6DOF actions
            pos = db['eef_pose'][i, :3, 3]  # (x, y, z)
            rpy = np.array(mat2euler(db['eef_pose'][i, :3, :3]))  # (roll, pitch, yaw)

            # calculate delta action to next step
            # use eef_pose[t+1] - eef_pose[t]
            if i < len(db['timestamps']) - 1:
                next_pos = db['eef_pose'][i + 1, :3, 3]
                next_rpy = np.array(mat2euler(db['eef_pose'][i + 1, :3, :3]))
                delta_pos = next_pos - pos
                delta_rpy = next_rpy - rpy
                delta_gripper = [1 if db['gripper_cmd'][i+1] <= 0 else -1]
            else:
                delta_pos = np.zeros(3)
                delta_rpy = np.zeros(3)
                delta_gripper = [1 if db['gripper_cmd'][i] <= 0 else -1]
            recorded_action = np.concatenate((delta_pos, delta_rpy, delta_gripper))

            # get predicted action
            action = get_vla_action(
                model,
                processor,
                "openvla",
                observation,
                prompt,
                unnorm_key,
                True
            )

            # Predict Action (7-DoF; un-normalize)
            print(f"Action:   {action})")
            print(f"Expected: {recorded_action})\n")

            # Display the image Press 'q' to exit
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # graph action values on a line graph
    #         if summary is None:
    #             summary = action
    #         else:
    #             summary = np.vstack((summary, action))

    finally:
        pass
    #     cv2.destroyAllWindows()

    #     # Create line graphs from summary
    #     plt.subplot(2, 2, 1)
    #     plt.plot(summary[:, 0])
    #     plt.plot(np.array(recoded_actions[:summary.shape[0]])[:, 0])
    #     plt.xlabel('Frame')
    #     plt.ylabel('Delta Value')
    #     plt.title('Action Values Over Time (x)')
    #     plt.legend(['VLA Output', 'Expected'])
    #     plt.ylim(-.5, .5)

    #     plt.subplot(2, 2, 2)
    #     plt.plot(summary[:, 1])
    #     plt.plot(np.array(recoded_actions[:summary.shape[0]])[:, 1])
    #     plt.xlabel('Frame')
    #     plt.ylabel('Delta Value')
    #     plt.title('Action Values Over Time (y)')
    #     plt.legend(['VLA Output', 'Expected'])
    #     plt.ylim(-.5, .5)

    #     plt.subplot(2, 2, 3)
    #     plt.plot(summary[:, 2])
    #     plt.plot(np.array(recoded_actions[:summary.shape[0]])[:, 2])
    #     plt.xlabel('Frame')
    #     plt.ylabel('Delta Value')
    #     plt.title('Action Values Over Time (z)')
    #     plt.legend(['VLA Output', 'Expected'])
    #     plt.ylim(-.5, .5)

    #     plt.subplot(2, 2, 4)
    #     plt.plot(summary[:, -1])
    #     plt.plot(1 - (np.array(recoded_actions[:summary.shape[0]])[:, -1] / 2))
    #     plt.xlabel('Frame')
    #     plt.ylabel('Delta Value')
    #     plt.title('Action Values Over Time (g)')
    #     plt.legend(['VLA Output', 'Expected'])
    #     plt.ylim(-0.01, 1.01)

    #     plt.tight_layout()
    #     plt.show()
if __name__ == "__main__":
    main(parser.parse_args())