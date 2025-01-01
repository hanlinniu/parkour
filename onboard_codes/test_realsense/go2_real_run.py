import rclpy
from rclpy.node import Node
from robot_data_ros_topic_test import UnitreeRos2Real

import os
import os.path as osp
import json
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# from rsl_rl import modules


def load_model(folder_path, filename):
    model_path = os.path.join(folder_path, filename)
    model = torch.load(model_path)  # Load the saved model
    return model


class ZeroActModel(torch.nn.Module):
    def __init__(self, angle_tolerance= 0.15, delta= 0.2):
        super().__init__()
        self.angle_tolerance = angle_tolerance
        self.delta = delta

    def forward(self, dof_pos):
        target = torch.zeros_like(dof_pos)
        diff = dof_pos - target
        diff_large_mask = torch.abs(diff) > self.angle_tolerance
        target[diff_large_mask] = dof_pos[diff_large_mask] \
            - self.delta * torch.sign(diff[diff_large_mask])
        return target
    
class Go2Node(UnitreeRos2Real):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_class_name= "Go2", **kwargs)
        self.loop_counter = 0
        self.infos = {}
        self.infos["depth"] = None
        self.device = torch.device('cuda:0')


    def register_models(self, stand_model, estimator_model, depth_encoder_model, depth_actor_model):
        self.stand_model = stand_model

        self.estimator_model = estimator_model
        self.depth_encoder_model = depth_encoder_model
        self.depth_actor_model = depth_actor_model
        
        self.use_stand_policy = True # Start with standing model

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
        
    def main_loop(self):
        if (self.joy_stick_buffer.keys & self.WirelessButtons.L1) and self.use_stand_policy:
            self.get_logger().info("L1 pressed, stop using stand policy")
            self.use_stand_policy = False
        if self.use_stand_policy:
            obs = self._get_dof_pos_obs() # do not multiply by obs_scales["dof_pos"]
            action = self.stand_model(obs)
            if (action == 0).all():
                self.get_logger().info("All actions are zero, it's time to switch to the policy", throttle_duration_sec= 1)
                # else:
                    # print("maximum dof error: {:.3f}".format(action.abs().max().item(), end= "\r"))
            self.send_action(action / self.action_scale)
        else:
            self.loop_counter += 1
            vision_obs = self._get_depth_obs()  # torch.Size([1, 58, 87])
            obs = self.read_observation()   # torch.Size([1, 753])

            if (self.loop_counter % 5 == 0) & (vision_obs is not None):
                self.infos["depth"] = vision_obs.clone()
            else: self.infos["depth"] = None

            if self.infos["depth"] is not None:
                obs_student = obs[:, :53].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = self.depth_encoder_model(self.infos["depth"], obs_student)  #  output torch.Size([1, 34])
                depth_latent = depth_latent_and_yaw[:, :-2]  # torch.Size([1, 34])
                yaw = depth_latent_and_yaw[:, -2:]  # torch.Size([1, 2])
            else:
                print("it is using depth camera, infos has no depth info")

            obs[:, 6:8] = 1.5*yaw
            obs_est = obs.clone()
            priv_states_estimated = self.estimator_model(obs_est[:, :53])         # output is 9, estimate velocity stuff
            obs_est[:, 53+132:53+132+9] = priv_states_estimated

            actions = self.depth_actor_model(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
            self.send_action(actions)
            
        if (self.joy_stick_buffer.keys & self.WirelessButtons.Y):
            self.get_logger().info("Y pressed, use the stand policy")
            self.use_stand_policy = True

            # start_time = time.monotonic()
            # obs = self.get_obs()
            # obs_time = time.monotonic()
            # action = self.task_policy(obs)
            # policy_time = time.monotonic()
            # self.send_action(action)
            # self.send_action(self._get_dof_pos_obs() / self.action_scale)
            # publish_time = time.monotonic()
            # print(
            #     "obs_time: {:.5f}".format(obs_time - start_time),
            #     "policy_time: {:.5f}".format(policy_time - obs_time),
            #     "publish_time: {:.5f}".format(publish_time - policy_time),
            # )


@torch.inference_mode()
def main(args):
    rclpy.init()

    save_folder = os.path.expanduser("~/learning-parkour/legged_gym/legged_gym/scripts/saved_models")
    estimator = load_model(folder_path=save_folder, filename="estimator.pth")
    depth_encoder = load_model(folder_path=save_folder, filename="depth_encoder.pth")
    depth_actor = load_model(folder_path=save_folder, filename="depth_actor.pth")

    # Print loaded models
    print("Loaded estimator is:", estimator)
    print("Loaded depth_encoder is:", depth_encoder)
    print("Loaded depth_actor is:", depth_actor)

    duration = 0.02  # for control frequency
    device = torch.device('cuda:0')

    env_node = Go2Node(
        "go2",
        model_device= device,
        dryrun= not args.nodryrun,
    )

    zero_act_model = ZeroActModel()
    zero_act_model = torch.jit.script(zero_act_model)

    env_node.register_models(
        zero_act_model, 
        estimator, 
        depth_encoder, 
        depth_actor
    )

    env_node.start_main_loop_timer(duration=0.02)
    rclpy.spin(env_node)
    rclpy.shuntdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")

    args = parser.parse_args()
    main(args)


