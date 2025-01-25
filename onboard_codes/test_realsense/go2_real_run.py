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

from rsl_rl import modules
from sport_api_constants import *
import time

def load_model(folder_path, filename):
    model_path = os.path.join(folder_path, filename)
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the saved model
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
        self.device = torch.device('cpu')

        save_data_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense/saved_data")
        self.flat_depth_data = load_model(folder_path=save_data_folder, filename="step_215_depth_data.pth")

        self.yaw = torch.zeros(1, 2, device=self.device)  # Initialize yaw to zeros
        self.depth_latent = torch.zeros(1, 32, device=self.device)  # Initialize depth_latent to zeros
        
    def register_models(self, stand_model, estimator_model, depth_encoder_model, depth_actor_model):
        self.stand_model = stand_model

        self.estimator_model = estimator_model
        self.depth_encoder_model = depth_encoder_model
        self.depth_actor_model = depth_actor_model
        
        self.use_stand_policy = False # Start with standing model
        self.use_parkour_policy = False
        self.use_sport_mode = True

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
        
    def main_loop(self):

        if self.use_sport_mode:
            if (self.joy_stick_buffer.keys & self.WirelessButtons.L1):
                self.get_logger().info("L1 pressed, Robot stand up")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.X):
                self.get_logger().info("X pressed, Robot balance stand")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.L2):
                self.get_logger().info("L2 pressed, Robot sit down")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R1):
                self.get_logger().info("R1 pressed, Switch to stand policy")
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_stand_policy = True
                self.use_parkour_policy = False

        if self.use_stand_policy:
            obs = self._get_dof_pos_obs() # do not multiply by obs_scales["dof_pos"]

            # obs_parkour = self.read_observation()   # torch.Size([1, 753])
            
            action = self.stand_model(obs)
            if (action == 0).all():
                self.get_logger().info("All actions are zero, it's time to switch to the policy", throttle_duration_sec= 1)
                # else:
                    # print("maximum dof error: {:.3f}".format(action.abs().max().item(), end= "\r"))
            # self.send_action(action / self.action_scale)
            self.send_action(action)

        if (self.joy_stick_buffer.keys & self.WirelessButtons.Y):
            self.get_logger().info("Y pressed, use the parkour policy")
            self.use_stand_policy = False
            self.use_parkour_policy = True
            self.use_sport_mode = False
            self.loop_counter = 0

        if self.use_parkour_policy:
            self.loop_counter += 1
            # vision_obs = self._get_depth_obs()  # torch.Size([1, 58, 87])
            self.vision_obs = self.flat_depth_data
            self.obs = self.read_observation()   # torch.Size([1, 753])

            if (self.loop_counter % 5 == 0) & (self.vision_obs is not None):
                self.infos["depth"] = self.vision_obs.clone()
            else: self.infos["depth"] = None

            if self.infos["depth"] is not None:
                self.obs_student = self.obs[:, :53].clone()
                self.obs_student[:, 6:8] = 0
                self.depth_latent_and_yaw = self.depth_encoder_model(self.infos["depth"], self.obs_student)  #  output torch.Size([1, 34])
                self.depth_latent = self.depth_latent_and_yaw[:, :-2]  # torch.Size([1, 32])
                self.yaw = self.depth_latent_and_yaw[:, -2:]  # torch.Size([1, 2])
                print("it is using depth camera, infos has depth info")
            else:
                print("it is using depth camera, infos has no depth info")

            self.obs[:, 6:8] = 1.5*self.yaw
            
            self.obs_est = self.obs.clone()
            self.priv_states_estimated = self.estimator_model(self.obs_est[:, :53])         # output is 9, estimate velocity stuff
            self.obs_est[:, 53+132:53+132+9] = self.priv_states_estimated

            self.actions = self.depth_actor_model(self.obs_est.detach(), hist_encoding=True, scandots_latent=self.depth_latent)
            # print("actions: ", actions)
            self.send_action(self.actions)

        if (self.joy_stick_buffer.keys & self.WirelessButtons.R2):
            if self.use_parkour_policy:
                self.get_logger().info("R2 pressed, stop using parkour policy, switch to sport mode")
            if self.use_stand_policy:
                self.get_logger().info("R2 pressed, stop using stand policy, switch to sport mode")
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self._sport_state_change(1)
            self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)



        # if (self.joy_stick_buffer.keys & self.WirelessButtons.L1) and self.use_stand_policy:
        #     self.get_logger().info("L1 pressed, stop using stand policy")
        #     self.use_stand_policy = False
        # if self.use_stand_policy:
        #     obs = self._get_dof_pos_obs() # do not multiply by obs_scales["dof_pos"]
        #     action = self.stand_model(obs)
        #     if (action == 0).all():
        #         self.get_logger().info("All actions are zero, it's time to switch to the policy", throttle_duration_sec= 1)
        #         # else:
        #             # print("maximum dof error: {:.3f}".format(action.abs().max().item(), end= "\r"))
        #     self.send_action(action / self.action_scale)
        # else:
        #     self.loop_counter += 1
        #     vision_obs = self._get_depth_obs()  # torch.Size([1, 58, 87])
        #     obs = self.read_observation()   # torch.Size([1, 753])

        #     if (self.loop_counter % 5 == 0) & (vision_obs is not None):
        #         self.infos["depth"] = vision_obs.clone()
        #     else: self.infos["depth"] = None

        #     if self.infos["depth"] is not None:
        #         obs_student = obs[:, :53].clone()
        #         obs_student[:, 6:8] = 0
        #         depth_latent_and_yaw = self.depth_encoder_model(self.infos["depth"], obs_student)  #  output torch.Size([1, 34])
        #         depth_latent = depth_latent_and_yaw[:, :-2]  # torch.Size([1, 34])
        #         yaw = depth_latent_and_yaw[:, -2:]  # torch.Size([1, 2])
        #     else:
        #         print("it is using depth camera, infos has no depth info")

        #     obs[:, 6:8] = 1.5*yaw
        #     obs_est = obs.clone()
        #     priv_states_estimated = self.estimator_model(obs_est[:, :53])         # output is 9, estimate velocity stuff
        #     obs_est[:, 53+132:53+132+9] = priv_states_estimated

        #     actions = self.depth_actor_model(obs_est.detach(), hist_encoding=True, scandots_latent=depth_latent)
        #     self.send_action(actions)
            
        # if (self.joy_stick_buffer.keys & self.WirelessButtons.Y):
        #     self.get_logger().info("Y pressed, use the stand policy")
        #     self.use_stand_policy = True




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

    save_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense/saved_models")
    save_data_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense/saved_data")

    estimator = load_model(folder_path=save_folder, filename="estimator.pth")
    depth_encoder = load_model(folder_path=save_folder, filename="depth_encoder.pth")
    depth_actor = load_model(folder_path=save_folder, filename="depth_actor.pth")

    flat_depth_data = load_model(folder_path=save_data_folder, filename="step_215_depth_data.pth")

    # Print loaded models
    print("Loaded estimator is:", estimator)
    print("Loaded depth_encoder is:", depth_encoder)
    print("Loaded depth_actor is:", depth_actor)
    print("flat_depth_data is:", flat_depth_data)

    duration = 0.02  # for control frequency
    device = torch.device('cpu')
    print("Models are loaded")
    env_node = Go2Node(
        "Go2",
        model_device= device,
        dryrun= not args.nodryrun,
    )
    print("Models are registered")

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
    parser.add_argument("--nodryrun", action= "store_true", default= True, help= "Disable dryrun mode")

    args = parser.parse_args()
    main(args)


