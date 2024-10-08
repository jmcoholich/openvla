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

# General
import torch
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import os
import json


def main():
    # Initialize robot
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

    easy_start = [ 0.04542551, 0.41048467, 0.0139709, -1.83497724, -0.14513091, 2.2443767, 1.06756333]
    reset_joints_to(robot_interface, reset_joint_positions)  # reset joints to home position
    reset_gripper(robot_interface, logger)  # reset gripper to open


    # Load Processor & VLA
    model_path = "/home/ripl/openvla/runs/openvla-7b+franka_pick_coke+b8+lr-2e-05+lora-r32+dropout-0.0+360x360"
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
    # unnorm_key = "dlr_edan_shared_control_converted_externally_to_rlds"  # cam near base angled towards table
    # unnorm_key = "nyu_franka_play_dataset_converted_externally_to_rlds"  # franka robot cam angled away from base
    # unnorm_key = "stanford_hydra_dataset_converted_externally_to_rlds"  # franka robot cam angled towards base
    # unnorm_key = "utaustin_mutex"  # franka robot cam straight on: this gives much bigger values than other keys so the robot moves fast
    task_label = "pick up the coke can"  # task to perform
    prompt = f"In: What action should the robot take to {task_label} ?\nOut:"

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
        while True:
            # Wait for a color frame
            frames = image_subscriber.recv_rgb_image()
            color_frame = frames[0]

            if color_frame is None:
                continue

            # breakpoint()
            color_frame = color_frame[:, 140:500]  # center crop 360x360
            # color_frame = color_frame[:, :, ::-1]  # BGR to RGB
            # color_frame = cv2.resize(color_frame[:, 140:500], (224, 224))  # make the image square then scale to 224x224

            # Convert the frame to PIL Image
            # image: Image.Image = Image.fromarray(color_frame)

            observation = {
                        "full_image": color_frame,
                        "state": np.concatenate(
                            (robot_interface.last_eef_quat_and_pos[1].flatten(),
                             quat2axisangle(robot_interface.last_eef_quat_and_pos[0]),
                             [robot_interface.last_gripper_q])
                        ),
                    }
            # breakpoint()
            # Query model to get action

            action = get_vla_action(
                model,
                processor,
                "openvla",
                observation,
                task_label,
                unnorm_key,
                center_crop=True
            )
            # action[:-1] = [i*10 for i in action[:-1]]

            # # Predict Action (7-DoF; un-normalize)
            # inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            # action = vla.predict_action(**inputs, unnorm_key="nyu_franka_play_dataset_converted_externally_to_rlds", do_sample=False)
            print(action)

            move_to_target_pose(
                robot_interface,
                controller_type,
                controller_cfg,
                target_delta_pose=action[:6],
                num_steps=1,
                num_additional_steps=0,
                interpolation_method="linear",
                )
            delta_gripper(robot_interface, logger, action[6])

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
        plt.plot(summary)
        plt.xlabel('Frame')
        plt.ylabel('Action Value')
        plt.title('Action Values Over Time')
        plt.legend(['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'])
        plt.show()

if __name__ == "__main__":
    main()