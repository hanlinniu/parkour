import os, sys

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    # MotorState,
    # IMUState,
    LowCmd,
    # MotorCmd,
)
from std_msgs.msg import Float32MultiArray


base_path = "/home/hanlin/parkour/onboard_codes/go2"
if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(os.path.join(base_path, "x86"))
elif os.uname().machine == "aarch64":
    sys.path.append(os.path.join(base_path, "aarch64"))

from crc_module import get_crc

from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import torch


class RobotCfgs:
    class Go2:
        NUM_DOF = 12
        NUM_ACTIONS = 12
        dof_map = [ # from isaacgym simulation joint order to real robot joint order
            3, 4, 5,
            0, 1, 2,
            9, 10, 11,
            6, 7, 8,
        ]
        dof_names = [ # NOTE: order matters. This list is the order in simulation.
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        dof_signs = [1.] * 12
        joint_limits_high = torch.tensor([
            1.0472, 3.4907, -0.83776,
            1.0472, 3.4907, -0.83776,
            1.0472, 4.5379, -0.83776,
            1.0472, 4.5379, -0.83776,
        ], device= "cpu", dtype= torch.float32)
        joint_limits_low = torch.tensor([
            -1.0472, -1.5708, -2.7227,
            -1.0472, -1.5708, -2.7227,
            -1.0472, -0.5236, -2.7227,
            -1.0472, -0.5236, -2.7227,
        ], device= "cpu", dtype= torch.float32)
        torque_limits = torch.tensor([ # from urdf and in simulation order
            25, 40, 40,
            25, 40, 40,
            25, 40, 40,
            25, 40, 40,
        ], device= "cpu", dtype= torch.float32)
        turn_on_motor_mode = [0x01] * 12


class UnitreeRos2Real(Node):
    """ A proxy implementation of the real H1 robot. """
    class WirelessButtons:
        R1 =            0b00000001 # 1
        L1 =            0b00000010 # 2
        start =         0b00000100 # 4
        select =        0b00001000 # 8
        R2 =            0b00010000 # 16
        L2 =            0b00100000 # 32
        F1 =            0b01000000 # 64
        F2 =            0b10000000 # 128
        A =             0b100000000 # 256
        B =             0b1000000000 # 512
        X =             0b10000000000 # 1024
        Y =             0b100000000000 # 2048
        up =            0b1000000000000 # 4096
        right =         0b10000000000000 # 8192
        down =          0b100000000000000 # 16384
        left =          0b1000000000000000 # 32768

    def __init__(self,
            robot_namespace= None,
            low_state_topic= "/lowstate",
            low_cmd_topic= "/lowcmd",
            joy_stick_topic= "/wirelesscontroller",
            forward_depth_topic= None, # if None and still need access, set to str "pyrealsense"
            forward_depth_embedding_topic= "/forward_depth_embedding",
            cfg= dict(),
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            cmd_px_range= [0.4, 1.0], # check joy_stick_callback (p for positive, n for negative)
            cmd_nx_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_py_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_ny_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_pyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            cmd_nyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            replace_obs_with_embeddings= [], # a list of strings, e.g. ["forward_depth"] then the corrseponding obs will be processed by _get_forward_depth_embedding_obs()
            move_by_wireless_remote= True, # if True, the robot will be controlled by a wireless remote
            model_device= "cpu",
            dof_pos_protect_ratio= 1.1, # if the dof_pos is out of the range of this ratio, the process will shutdown.
            robot_class_name= "Go2",
            dryrun= True, # if True, the robot will not send commands to the real robot
        ):
        super().__init__("unitree_ros2_real")
        self.NUM_DOF = getattr(RobotCfgs, robot_class_name).NUM_DOF
        self.NUM_ACTIONS = getattr(RobotCfgs, robot_class_name).NUM_ACTIONS
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic