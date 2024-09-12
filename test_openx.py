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
import h5py

# General
import torch
import numpy as np
from time import sleep
import matplotlib.pyplot as plt


def main():
    # load data
    # Load example data and print data structure


    data = {}
    images = []
    recoded_actions = []
    prompts = []
    with h5py.File("/home/ripl/openvla/all_play_data_diverse/all_play_data_diverse.h5", "r") as F:
        # breakpoint()
        for i in range(146, F["terminals"].size):
            images.append(F["front_cam_ob"][i])
            recoded_actions.append(F["actions"][i])
            prompts.append(F["prompts"][i])

            if F["terminals"][i]:
                break



    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",  # should use all available GPUs
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    unnorm_key = "jaco_play"  # cam near base angled towards table
    # task_label = "pick coke can"
    # prompt = f"In: What action should the robot take to {task_label} ?\nOut:"

    # np array for storing action values
    summary = None

    # Main loop
    try:
        for i, frame in enumerate(images):
            # Convert the frame to PIL Image
            # video_frame = cv2.resize(color_frame, (224, 224))
            image: Image.Image = Image.fromarray(frame)

            observation = {
                        "full_image": frame,
                        "state": None,
                    }
            # breakpoint()
            # Query model to get action

            action = get_vla_action(
                model,
                processor,
                "openvla",
                observation,
                prompts[i],
                unnorm_key
            )

            # Predict Action (7-DoF; un-normalize)
            print(f"Action:   {action})")
            print(f"Expected: {recoded_actions[i]})\n")

            # Display the image Press 'q' to exit
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # graph action values on a line graph
            if summary is None:
                summary = action
            else:
                summary = np.vstack((summary, action))

    finally:
        cv2.destroyAllWindows()

        # Create line graphs from summary
        plt.subplot(2, 2, 1)
        plt.plot(summary[:, 0])
        plt.plot(np.array(recoded_actions[:summary.shape[0]])[:, 0])
        plt.xlabel('Frame')
        plt.ylabel('Delta Value')
        plt.title('Action Values Over Time (x)')
        plt.legend(['VLA Output', 'Expected'])
        plt.ylim(-.5, .5)

        plt.subplot(2, 2, 2)
        plt.plot(summary[:, 1])
        plt.plot(np.array(recoded_actions[:summary.shape[0]])[:, 1])
        plt.xlabel('Frame')
        plt.ylabel('Delta Value')
        plt.title('Action Values Over Time (y)')
        plt.legend(['VLA Output', 'Expected'])
        plt.ylim(-.5, .5)

        plt.subplot(2, 2, 3)
        plt.plot(summary[:, 2])
        plt.plot(np.array(recoded_actions[:summary.shape[0]])[:, 2])
        plt.xlabel('Frame')
        plt.ylabel('Delta Value')
        plt.title('Action Values Over Time (z)')
        plt.legend(['VLA Output', 'Expected'])
        plt.ylim(-.5, .5)

        plt.subplot(2, 2, 4)
        plt.plot(summary[:, -1])
        plt.plot(1 - (np.array(recoded_actions[:summary.shape[0]])[:, -1] / 2))
        plt.xlabel('Frame')
        plt.ylabel('Delta Value')
        plt.title('Action Values Over Time (g)')
        plt.legend(['VLA Output', 'Expected'])
        plt.ylim(-0.01, 1.01)

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()