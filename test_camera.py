''' script to grab camera feed and store as PIL Image'''


import cv2
from PIL import Image
import pyrealsense2 as rs
import numpy as np
from openteach.utils.network import ZMQCameraSubscriber


# Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
# pipeline.start(config)

image_subscriber = ZMQCameraSubscriber(
            host = "143.215.128.151",
            port = "10007",
            topic_type = 'RGB'
        )

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = image_subscriber.recv_rgb_image()
        # breakpoint()
        color_frame = frames[0]
        # breakpoint()
        if color_frame is None:
            continue

        # Convert the frame to PIL Image
        # img = np.asanyarray(color_frame.as_frame().get_data())
        # img = cv2.resize(img, (224, 224))
        image: Image.Image = Image.fromarray(color_frame)
        # Display the image
        cv2.imshow("Camera", color_frame)
        # breakpoint()

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    # pipeline.stop()
    cv2.destroyAllWindows()


