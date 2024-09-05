''' Use openVLA with camera feed, TODO: move robot with outputs'''

# Camera
import cv2
from PIL import Image
from openteach.utils.network import ZMQCameraSubscriber

# openVLA
from transformers import AutoModelForVision2Seq, AutoProcessor

# deoxys_control
from deoxys.franka_interface import FrankaInterface
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
from deoxys.utils.config_utils import get_default_controller_config
from deoxys_control.osc_control import move_to_target_pose, reset_joints_to  # borrowing this to move based on deltas
from deoxys_control.delta_gripper import delta_gripper, reset_gripper
from deoxys.utils.log_utils import get_deoxys_example_logger

# General
import torch
import numpy as np
from time import sleep
import matplotlib.pyplot as plt


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
    reset_joints_to(robot_interface, reset_joint_positions)  # reset joints to home position
    reset_gripper(robot_interface, logger)  # reset gripper to open


    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",  # should use all available GPUs
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Configure Camera Stream
    image_subscriber = ZMQCameraSubscriber(
            host = "143.215.128.151",
            port = "10007",
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

            # Convert the frame to PIL Image
            # video_frame = cv2.resize(video_frame, (224, 224))
            image: Image.Image = Image.fromarray(color_frame)
            prompt = "In: What action should the robot take to {pick up the stuffed animal}?\nOut:"

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            action = vla.predict_action(**inputs, unnorm_key="nyu_franka_play_dataset_converted_externally_to_rlds", do_sample=False)  # action[:6] = pose, action[6] = gripper
            print(action)

            move_to_target_pose(
                robot_interface,
                controller_type,
                controller_cfg,
                target_delta_pose=action[:6],
                num_steps=5,
                num_additional_steps=0,
                interpolation_method="linear",
                )
            delta_gripper(robot_interface, logger, action[6])

            # Display the image Press 'q' to exit
            cv2.imshow("Camera", color_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # graph action values on a line graph
            if summary is None:
                summary = action
            else:
                summary = np.vstack((summary, action))

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