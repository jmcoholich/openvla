''' Use openVLA with camera feed, TODO: move robot with outputs'''

# openVLA
from transformers import AutoModelForVision2Seq, AutoProcessor
from experiments.robot.robot_utils import get_vla_action, normalize_gripper_action
from prismatic.vla.action_tokenizer import ActionTokenizer


# General
import torch
import os
import json
import tensorflow_datasets as tfds
import cv2



def main():
    prompt = "pick up the coke can"
    # module = importlib.import_module("franka_pick_coke")
    ds = tfds.load("franka_pick_coke_rgb", split='train')

    # Load Processor & VLA
    model_path = "/home/ripl/openvla/runs/openvla-7b+franka_pick_coke+RGB+Euler+cmd_gripper"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # should use all available GPUs
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)

    model.norm_stats = norm_stats
    unnorm_key = "franka_pick_coke"

    accuracy = 0

    # Main loop
    try:
        for i, episode in enumerate(ds.take(1)):
            for j, st in enumerate(episode['steps']):
                frame = st['observation']['image'].numpy()
                frame = cv2.resize(frame, (224, 224))  # for some reason the image processor expects 224x224 not 360x360
                observation = {
                            "full_image": frame,
                            "state": None,
                        }

                recorded_action = st['action'].numpy()  # the action are deltas for (x, y, z, r, p, y, gripper)

                # get predicted action
                action = get_vla_action(
                    model,
                    processor,
                    "openvla",
                    observation,
                    prompt,
                    unnorm_key,
                    False
                )

                action = normalize_gripper_action(action, binarize=True)
                recorded_action = normalize_gripper_action(recorded_action, binarize=True)



                # Predict Action (7-DoF; un-normalize)
                print(f"Predicted: {action}")
                print(f"Recorded:  {recorded_action}")
                print(f"Match:   {action_tokenizer(action) == action_tokenizer(recorded_action)}")
                accuracy = accuracy + (1 if action_tokenizer(action) == action_tokenizer(recorded_action) else 0)
                print(f"Accuracy: {accuracy / (j + 1)}")

    finally:
        pass


if __name__ == "__main__":
    main()