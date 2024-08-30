# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
import numpy as np

from time import sleep

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


# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    # load_in_4bit=True,
    device_map="auto",  # should use all available GPUs
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
#.to("cuda:0")
# ).to("cpu")

# load still_img.png to use as test image
image: Image.Image =  Image.open("still_img.png")


# Grab image input & format prompt
# image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {pick up the stuffed animal}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)  # action[:6] = pose, action[6] = gripper
# action = delta [x, y, z, roll, pitch, yaw, gripper]
# Execute...

breakpoint()

