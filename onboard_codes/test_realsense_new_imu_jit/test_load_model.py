# import rclpy
# from rclpy.node import Node
# from robot_data_ros_topic_test import UnitreeRos2Real

import os
import os.path as osp
import json
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from rsl_rl.modules import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87

from rsl_rl import modules
from sport_api_constants import *
import time



def load_model(folder_path, filename):
    model_path = os.path.join(folder_path, filename)
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the saved model
    return model



@torch.inference_mode()

def main(args):

    device = "cuda"

    save_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense_new_imu_jit/saved_models")
    
    base_model = torch.jit.load(os.path.join(save_folder, "0121-distill-policy-mlp-model-27000-base_jit.pt"), map_location=device)
    base_model.eval()
    
    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone

    print("*"*80)
    print("actor modules are: ")
    # Get all immediate children (not recursive) under actor
    for name, module in base_model.actor.named_children():
        if name not in ["history_encoder", "actor_backbone"]:
            print(f"{name}: {module}")


    import inspect

    try:
        forward_fn = base_model.actor.forward
        sig = inspect.signature(forward_fn)
        print(f"Signature: {sig}")
        args = [p for p in sig.parameters.values() if p.name != "self"]
        print(f"Number of arguments: {len(args)}")
        print("Argument names:", [a.name for a in args])
    except ValueError as e:
        print("Could not inspect signature. Likely due to JIT tracing or scripting.")
        print("Error:", e)

    print("*"*80)

    vision_model = torch.load(os.path.join(save_folder, "0121-distill-policy-mlp-model-27000-vision_weight.pt"), map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()

    print("*"*80)
    print("estimator type:", type(estimator))
    print("estimator is:", estimator)
    print("depth_encoder type:", type(depth_encoder))
    print("depth_encoder is:", depth_encoder)




    estimator1 = load_model(folder_path=save_folder, filename="estimator.pth")
    # depth_encoder = load_model(folder_path=save_folder, filename="depth_encoder.pth")
    # depth_actor = load_model(folder_path=save_folder, filename="depth_actor.pth")
    print("estimator1 type:", type(estimator1))
    print("estimator1 is:", estimator1)



    # whole_model = torch.load(os.path.join(save_folder, "model_27000.pt"), map_location='cpu')
    # estimator_whole = whole_model["estimator_state_dict"]
    # depth_encoder_whole = whole_model["depth_encoder_state_dict"]
    # depth_actor_whole = whole_model["depth_actor_state_dict"]

    # print("depth_encoder type:", type(depth_encoder))
    # print("estimator type:", type(estimator))
    # print("estimator_whole type:", type(estimator_whole))


    # whole_model = torch.jit.load(os.path.join(save_folder, "0121-distill-policy-mlp-model-27000-base_jit.pt"), map_location='cpu')
    # whole_model.eval()
    # estimator = whole_model.estimator.estimator

    # for k in estimator_whole:
    #     if torch.is_tensor(estimator_whole[k]):
    #         estimator_whole[k] = estimator_whole[k] + 0.01


    # # Print loaded models
    # print("Loaded estimator is:", estimator)
    # print("Loaded depth_encoder is:", depth_encoder)
    # print("Loaded depth_actor is:", depth_actor)
    # # print("Loaded whole_model is:", whole_model)
    
    # if isinstance(whole_model, dict):
    #     print("Number of items in whole_model:", len(whole_model))
    #     print("Keys:", list(whole_model.keys()))

    # # Check if keys are the same
    # keys_match = list(estimator.state_dict().keys()) == list(estimator_whole.keys())
    # print("Do keys match:", keys_match)

    # # Check if all parameter tensors match
    # params_match = all(torch.equal(estimator.state_dict()[k].cpu(), estimator_whole[k].cpu())
    #                 for k in estimator.state_dict() if k in estimator_whole)
    # print("Do all parameter tensors match:", params_match)


    # # Check depth_encoder
    # depth_encoder_keys_match = list(depth_encoder.state_dict().keys()) == list(depth_encoder_whole.keys())
    # print("Do depth_encoder keys match:", depth_encoder_keys_match)

    # depth_encoder_values_match = all(
    #     torch.equal(depth_encoder.state_dict()[k].cpu(), depth_encoder_whole[k].cpu())
    #     for k in depth_encoder.state_dict() if k in depth_encoder_whole
    # )
    # print("Do depth_encoder parameter tensors match:", depth_encoder_values_match)

    # # Check depth_actor
    # depth_actor_keys_match = list(depth_actor.state_dict().keys()) == list(depth_actor_whole.keys())
    # print("Do depth_actor keys match:", depth_actor_keys_match)

    # depth_actor_values_match = all(
    #     torch.equal(depth_actor.state_dict()[k].cpu(), depth_actor_whole[k].cpu())
    #     for k in depth_actor.state_dict() if k in depth_actor_whole
    # )
    # print("Do depth_actor parameter tensors match:", depth_actor_values_match)


    # # Check if keys are the same
    # keys_match = list(estimator.state_dict().keys()) == list(estimator1.state_dict().keys())
    # print("Do keys match:", keys_match)

    # # Check if all parameter tensors match
    # params_match = all(torch.equal(estimator.state_dict()[k].cpu(), estimator1.state_dict()[k].cpu())
    #                 for k in estimator.state_dict() if k in estimator1.state_dict())
    # print("Do all parameter tensors match:", params_match)



    # # Check if keys are the same
    # # 1. Check keys first
    # model_keys = set(estimator.state_dict().keys())
    # whole_keys = set(estimator_whole.keys())

    # missing_in_whole = model_keys - whole_keys
    # missing_in_model = whole_keys - model_keys

    # if missing_in_whole or missing_in_model:
    #     print("⚠️ Key mismatch detected!")
    #     if missing_in_whole:
    #         print("Keys missing in whole_model:", missing_in_whole)
    #     if missing_in_model:
    #         print("Keys missing in model:", missing_in_model)
    # else:
    #     print("✅ Keys fully match.")

    #     # 2. Compare values only if keys match
    #     values_match = all(torch.equal(estimator.state_dict()[k].cpu(), estimator_whole[k].cpu())
    #                     for k in model_keys)
    #     print("✅ Do all parameter tensors match:", values_match)


    # print("Loaded estimator_whole is:", estimator_whole)
    # print("Loaded depth_encoder_whole is:", depth_encoder_whole)
    # print("Loaded depth_actor_whole is:", depth_actor_whole)


    # estimator.eval()
    # depth_encoder.eval()
    # depth_actor.eval()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodryrun", action= "store_true", default= True, help= "Disable dryrun mode")

    args = parser.parse_args()
    main(args)


