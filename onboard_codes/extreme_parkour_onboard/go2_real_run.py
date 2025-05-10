import rclpy
from rclpy.node import Node
# from robot_data_ros_topic_test import UnitreeRos2Real
from unitree_ros2_real import UnitreeRos2Real, get_euler_xyz

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
        self.device = torch.device('cuda')

        # save_data_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense/saved_data")
        # self.flat_depth_data = load_model(folder_path=save_data_folder, filename="step_215_depth_data.pth")

        self.yaw = torch.zeros(1, 2, device=self.device)  # Initialize yaw to zeros
        self.depth_latent = torch.zeros(1, 32, device=self.device)  # Initialize depth_latent to zeros

        self.yaw_warmup = torch.zeros(1, 2, device=self.device)  # Initialize yaw to zeros
        self.depth_latent_warmup = torch.zeros(1, 32, device=self.device)

        self.yaw_log = []
        self.log_entry = []

        self.sim_ite = 3
        self.global_counter = 0
        self.visual_update_interval = 5

        
    def register_models(self, stand_model, turn_obs, depth_encode, policy):
        self.stand_model = stand_model
        self.turn_obs = turn_obs
        self.depth_encode = depth_encode
        self.policy = policy
        
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
        for _ in range(2):
            start_time = time.monotonic()

            proprio = self.get_proprio()
            get_pro_time = time.monotonic()
            proprio_history = self._get_history_proprio() 
            get_hist_pro_time = time.monotonic()

            if self.global_counter % self.visual_update_interval == 0:
                depth_image = self._get_depth_image()
                self.depth_latent_yaw = self.depth_encode(depth_image, proprio)

            get_obs_time = time.monotonic()

            obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, self.n_proprio, self.n_depth_latent, self.n_hist_len)

            turn_obs_time = time.monotonic()

            action = self.policy(obs)
            policy_time = time.monotonic()

            publish_time = time.monotonic()
            print("warm up: ",
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
                "policy_time: {:.5f}".format(policy_time - turn_obs_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )
        
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
            # obs_balance = self.read_observation()

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
            
            self.use_stand_policy = False
            self.use_sport_mode = False

            # self.vision_obs = self._get_depth_obs()  # torch.Size([1, 58, 87])
            # self.vision_obs = self.flat_depth_data

            print("*"*50)
            print("now it is in parkour policy")
            start_time = time.monotonic()

            proprio = self.get_proprio()
            get_pro_time = time.monotonic()

            proprio_history = self._get_history_proprio()
            get_hist_pro_time = time.monotonic()

            # print('proprioception: ', proprio)
            # print('history proprioception: ', proprio_history)

            if self.global_counter % self.visual_update_interval == 0:
                depth_image = self._get_depth_image()
                if self.global_counter == 0:
                    self.last_depth_image = depth_image
                self.depth_latent_yaw = self.depth_encode(self.last_depth_image, proprio)
                self.last_depth_image = depth_image
                # print('depth latent: ', self.depth_latent_yaw)
            get_obs_time = time.monotonic()

            obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, self.n_proprio, self.n_depth_latent, self.n_hist_len)
            turn_obs_time = time.monotonic()

            action = self.policy(obs)
            policy_time = time.monotonic()
            # print('action before clip and normalize: ', action)

            # action = self.actions_sim[self.sim_ite, :]
            self.send_action(action)
            # print('action: ', action)
            self.sim_ite += 1

            publish_time = time.monotonic()
            print(
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
                "policy_time: {:.5f}".format(policy_time - turn_obs_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )

            self.global_counter += 1

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


    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    
    config_dict["control"]["computer_clip_torque"] = True
    
    # duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # different from parkour
    device = "cuda"
    duration = 0.02

    env_node = Go2Node(
        "go2",
        cfg= config_dict,
        model_device= device,
        dryrun= not args.nodryrun,
    )

    env_node.get_logger().info("Model loaded from: {}".format(osp.join(args.logdir)))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))


    save_folder = os.path.expanduser("~/parkour/onboard_codes/test_realsense_new_imu_jit/saved_models")

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


    zero_act_model = ZeroActModel()
    zero_act_model = torch.jit.script(zero_act_model)
    zero_act_model.to(device)
    zero_act_model.eval()

    def turn_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        # yaw = depth_latent_yaw[:, -2:] * 0
        print('yaw: ', yaw)

        lin_vel_latent = estimator(proprio)

        activation = nn.ELU()
        priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))

        proprio[:, 6:8] = yaw
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)

        return obs

    def encode_depth(depth_image, proprio):
        depth_latent_yaw = depth_encoder(depth_image, proprio)
        if torch.isnan(depth_latent_yaw).any():
            print('depth_latent_yaw contains nan and the depth image is: ', depth_image)
        return depth_latent_yaw
    
    def actor_model(obs):
        action = actor(obs)
        return action
    
    def stand_model(obs):
        action = zero_act_model(obs)
        return action
    

    env_node.register_models(stand_model=stand_model, turn_obs=turn_obs, depth_encode=encode_depth, policy=actor_model)

    env_node.start_ros_handlers()
    env_node.warm_up()
    env_node.start_main_loop_timer(duration=0.02)
    rclpy.spin(env_node)
    rclpy.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )


    args = parser.parse_args()
    main(args)


