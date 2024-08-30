''' script to grab camera feed and store as PIL Image'''


import cv2
from PIL import Image
import pyrealsense2 as rs
import numpy as np


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert the frame to PIL Image
        # depth = frames.get_depth_frame()
        # depth_data = depth.as_frame().get_data()
        # np_image = np.asanyarray(depth_data)

        image: Image.Image = Image.fromarray(np.asanyarray(color_frame.as_frame().get_data()))
        # Display the image
        # cv2.imshow("Camera", image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()


