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
from torch import nn
from rsl_rl.modules import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87
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
        self.loop_counter_warmup = 0
        self.infos = {}
        self.infos["depth"] = None
        self.device = torch.device('cpu')

        # save_data_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense/saved_data")
        # self.flat_depth_data = load_model(folder_path=save_data_folder, filename="step_215_depth_data.pth")

        self.yaw = torch.zeros(1, 2, device=self.device)  # Initialize yaw to zeros
        self.depth_latent = torch.zeros(1, 32, device=self.device)  # Initialize depth_latent to zeros

        self.yaw_warmup = torch.zeros(1, 2, device=self.device)  # Initialize yaw to zeros
        self.depth_latent_warmup = torch.zeros(1, 32, device=self.device)

        self.yaw_log = []
        self.log_entry = []
        
    def register_models(self, stand_model, estimator_model, depth_encoder_model, depth_actor_model):
        self.stand_model = stand_model

        self.estimator_model = estimator_model
        self.depth_encoder_model = depth_encoder_model
        self.depth_actor_model = depth_actor_model
        
        self.use_stand_policy = False # Start with standing model
        self.use_parkour_policy = False
        self.use_sport_mode = True

        self.balance_policy_mode = False

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )

    def warm_up(self):
        for _ in range(5):
            start_time = time.monotonic()

            obs = self.read_observation()   # torch.Size([1, 753])
            get_obs_time = time.monotonic()

            vision_obs = self._get_depth_obs() 
            get_vision_time = time.monotonic()

            if (self.loop_counter_warmup % 5 == 0) & (vision_obs is not None):
                self.infos["depth"] = vision_obs.clone()
            else: self.infos["depth"] = None

            if self.infos["depth"] is not None:
                # print("self.loop_counter is ", self.loop_counter)
                print("depth image is here!")
                # print("self.infosdepth is ", self.infos["depth"][:, -10:, -10:])
                obs_student = obs[:, :53].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = self.depth_encoder_model(self.infos["depth"], obs_student)  #  output torch.Size([1, 34])
                self.depth_latent_warmup = depth_latent_and_yaw[:, :-2]  # torch.Size([1, 32])
                self.yaw_warmup = depth_latent_and_yaw[:, -2:]  # torch.Size([1, 2])
                print("it is using depth camera, infos has depth info")
            else:
                print("it is using depth camera, infos has no depth info")

            get_vision_latent_time = time.monotonic()
            obs[:, 6:8] = 1.5*self.yaw_warmup

            obs_est = obs.clone()
            priv_states_estimated = self.estimator_model(obs_est[:, :53])         # output is 9, estimate velocity stuff
            obs_est[:, 53+132:53+132+9] = priv_states_estimated.clone()

            get_obs_est_time = time.monotonic()

            actions = self.depth_actor_model(self.obs_est.detach(), hist_encoding=True, scandots_latent=self.depth_latent_warmup)
            publish_time = time.monotonic()

            print("*"*50)
            print("warm up: ",
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "get vision time: {:.5f}".format(get_vision_time - get_obs_time),
                "get vision latent time: {:.5f}".format(get_vision_latent_time - get_vision_time),
                "get obs est time: {:.5f}".format(get_obs_est_time - start_time),
                "policy_time: {:.5f}".format(publish_time- get_obs_est_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )
            self.loop_counter_warmup += 1   
        
    def main_loop(self):

        if self.use_sport_mode:
            if (self.joy_stick_buffer.keys & self.WirelessButtons.L1):
                self.get_logger().info("L1 pressed, Robot stand up")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.X):
                self.get_logger().info("X pressed, Robot balance stand")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)
                self.balance_policy_mode = True
            if (self.joy_stick_buffer.keys & self.WirelessButtons.L2):
                self.get_logger().info("L2 pressed, Robot sit down")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)
                self.balance_policy_mode = False
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R1):
                self.get_logger().info("R1 pressed, Switch to stand policy")
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_stand_policy = True
                self.use_parkour_policy = False
                self.balance_policy_mode = False

        if self.balance_policy_mode:
            self.get_logger().info("X pressed, recordning balance mode data")
            obs_balance = self.read_observation()

        if self.use_stand_policy:
            obs = self._get_dof_pos_obs() # do not multiply by obs_scales["dof_pos"]
            obs_parkour = self.read_observation()   # torch.Size([1, 753])
            
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
            
            self.use_stand_policy = False
            self.use_sport_mode = False

            # self.vision_obs = self._get_depth_obs()  # torch.Size([1, 58, 87])
            # self.vision_obs = self.flat_depth_data
            self.obs = self.read_observation()   # torch.Size([1, 753])

            if self.loop_counter % 5 == 0:
                vision_obs = self._get_depth_obs()  # torch.Size([1, 58, 87])
                if self.loop_counter == 0:
                    self.last_vision_obs = vision_obs.clone()
                self.obs_student = self.obs[:, :53].clone()
                self.depth_latent_and_yaw = self.depth_encoder_model(self.last_vision_obs, self.obs_student)  #  output torch.Size([1, 34])
                self.depth_latent = self.depth_latent_and_yaw[:, :-2]  # torch.Size([1, 32])
                self.yaw = self.depth_latent_and_yaw[:, -2:]  # torch.Size([1, 2])
                self.last_depth_image = vision_obs.clone()
            self.obs[:, 6:8] = 1.5*self.yaw


            ####################################################################
            ##########################log yaw data##############################
            # self.log_entry = [self.step_count,
            #              self.obs[:, 6].item(),
            #              self.obs[:, 7].item()]
            # self.yaw_log.append(self.log_entry)

            # if self.step_count % 20 == 0:
            #     save_path = os.path.expanduser("~/parkour/plot/yaw_log.npy")
            #     np.save(save_path, np.array(self.yaw_log)) # shape: (step, 11)
            ####################################################################
            
            
            self.obs_est = self.obs.clone()
            self.priv_states_estimated = self.estimator_model(self.obs_est[:, :53])         # output is 9, estimate velocity stuff
            self.obs_est[:, 53+132:53+132+9] = self.priv_states_estimated.clone()

            self.actions = self.depth_actor_model(self.obs_est.detach(), hist_encoding=True, scandots_latent=self.depth_latent)
            self.send_action(self.actions)

            self.loop_counter += 1

        if (self.joy_stick_buffer.keys & self.WirelessButtons.R2):
            if self.use_parkour_policy:
                self.get_logger().info("R2 pressed, stop using parkour policy, switch to sport mode")

            if self.use_stand_policy:
                self.get_logger().info("R2 pressed, stop using stand policy, switch to sport mode")
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self.yaw = torch.zeros(1, 2, device=self.device)  # Initialize yaw to zeros
            self.depth_latent = torch.zeros(1, 32, device=self.device)  # Initialize depth_latent to zeros
            self.reset_obs_buffers()
            self._sport_state_change(1)
            self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)



@torch.inference_mode()

def main(args):
    rclpy.init()

    save_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense_new_imu_jit/saved_models")
    device = "cuda"
    # device = torch.device('cpu')
    duration = 0.02  # for control frequency

    base_model = torch.jit.load(os.path.join(save_folder, "0121-distill-policy-mlp-model-27000-base_jit.pt"), map_location=device)
    base_model.eval()
    
    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone

    vision_model = torch.load(os.path.join(save_folder, "0121-distill-policy-mlp-model-27000-vision_weight.pt"), map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()

    print("Models are loaded")
    env_node = Go2Node(
        "Go2",
        model_device= device,
        dryrun= not args.nodryrun,
    )
    

    zero_act_model = ZeroActModel()
    zero_act_model = torch.jit.script(zero_act_model)

    env_node.register_models(
        zero_act_model, 
        estimator, 
        depth_encoder, 
        actor
    )
    print("Models are registered")



    env_node.start_ros_handlers()
    env_node.warm_up()
    env_node.start_main_loop_timer(duration=0.02)
    rclpy.spin(env_node)
    rclpy.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodryrun", action= "store_true", default= True, help= "Disable dryrun mode")

    args = parser.parse_args()
    main(args)


