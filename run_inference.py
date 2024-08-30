''' Use openVLA with camera feed, TODO: move robot with outputs'''

# Camera
import cv2
from PIL import Image
import pyrealsense2 as rs

# openVLA
from transformers import AutoModelForVision2Seq, AutoProcessor

# General
import torch
import numpy as np
from time import sleep


def main():
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",  # should use all available GPUs
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    pipeline.start(config)


    # Main loop
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image: Image.Image = Image.fromarray(np.asanyarray(color_frame.as_frame().get_data()))

            prompt = "In: What action should the robot take to {pick up the stuffed animal}?\nOut:"

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)  # action[:6] = pose, action[6] = gripper
            print(action)
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()