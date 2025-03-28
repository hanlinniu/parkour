import os, sys

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    SportModeState,
    # MotorState,
    # IMUState,
    LowCmd,
    # MotorCmd,
)
from unitree_api.msg import Request, RequestHeader

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo

base_path = "/home/unitree/parkour/onboard_codes/go2"
if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(os.path.join(base_path, "x86"))
elif os.uname().machine == "aarch64":
    sys.path.append(os.path.join(base_path, "aarch64"))

from crc_module import get_crc

from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import torch
from cv_bridge import CvBridge


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

        default_joint_angles = {
                    "FL_hip_joint": 0.1,
                    "FL_thigh_joint": 0.7,
                    "FL_calf_joint": -1.5,

                    "FR_hip_joint": -0.1,
                    "FR_thigh_joint": 0.7,
                    "FR_calf_joint": -1.5,

                    "RL_hip_joint": 0.1,
                    "RL_thigh_joint": 1.0,
                    "RL_calf_joint": -1.5,

                    "RR_hip_joint": -0.1,
                    "RR_thigh_joint": 1.0,
                    "RR_calf_joint": -1.5
                }
        
        stiffness = 40.0
        damping = 1.0
        

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
            sport_mode_state_topic = "/sportmodestate",
            low_cmd_topic= "/lowcmd",
            joy_stick_topic= "/wirelesscontroller",
            forward_depth_topic= "/camera/forward_depth", # if None and still need access, set to str "pyrealsense"
            forward_depth_embedding_topic= None,  # "/forward_depth_embedding",
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
            dof_pos_protect_ratio= 1.2, # if the dof_pos is out of the range of this ratio, the process will shutdown.
            robot_class_name= "Go2",
            dryrun= True, # if True, the robot will not send commands to the real robot
        ):
        super().__init__("unitree_ros2_real")
        self.NUM_DOF = getattr(RobotCfgs, robot_class_name).NUM_DOF
        self.NUM_ACTIONS = getattr(RobotCfgs, robot_class_name).NUM_ACTIONS
        self.robot_namespace = robot_namespace
        self.robot_class_name = robot_class_name
        self.low_state_topic = low_state_topic
        self.sport_mode_state_topic = sport_mode_state_topic
        self.low_cmd_topic = low_cmd_topic if not dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        self.joy_stick_topic = joy_stick_topic
        self.forward_depth_topic = forward_depth_topic
        self.forward_depth_embedding_topic = forward_depth_embedding_topic
        self.dryrun = dryrun

        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.cmd_px_range = cmd_px_range
        self.cmd_nx_range = cmd_nx_range
        self.cmd_py_range = cmd_py_range
        self.cmd_ny_range = cmd_ny_range
        self.cmd_pyaw_range = cmd_pyaw_range
        self.cmd_nyaw_range = cmd_nyaw_range

        self.replace_obs_with_embeddings = replace_obs_with_embeddings
        self.move_by_wireless_remote = move_by_wireless_remote
        self.model_device = model_device
        self.dof_pos_protect_ratio = dof_pos_protect_ratio

        self.dof_map = getattr(RobotCfgs, self.robot_class_name).dof_map
        self.dof_names = getattr(RobotCfgs, self.robot_class_name).dof_names
        self.dof_signs = getattr(RobotCfgs, self.robot_class_name).dof_signs
        self.turn_on_motor_mode = getattr(RobotCfgs, self.robot_class_name).turn_on_motor_mode
        self.latest_sportmodestate_msg = None

        self.clip_obs = 100.0
        self.step_count = 0
        self.last_contacts = [False, False, False, False]
        self.contact_filt = [False, False, False, False]

        self.bridge = CvBridge()

        # for control frequency
        self.dt = 0.002
        self.decimation = 4
        self.action_scale = 0.25
        self.clip_actions = 4.8
        self.torque_limits = getattr(RobotCfgs, self.robot_class_name).torque_limits.to(self.model_device)

        self.parse_config()
        self.start_ros_handlers()

        self.data_log = []

    def parse_config(self):

        self.p_gains = getattr(RobotCfgs, self.robot_class_name).stiffness
        self.d_gains = getattr(RobotCfgs, self.robot_class_name).damping

        self.default_dof_pos = torch.zeros(self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_pos_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_vel_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)

        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            default_joint_angle = getattr(RobotCfgs, self.robot_class_name).default_joint_angles[name]
            # in simulation order.
            self.default_dof_pos[i] = default_joint_angle

        
        # actions
        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)

        # hardware related, in simulation order
        self.joint_limits_high = getattr(RobotCfgs, self.robot_class_name).joint_limits_high.to(self.model_device)
        self.joint_limits_low = getattr(RobotCfgs, self.robot_class_name).joint_limits_low.to(self.model_device)
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.dof_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.dof_pos_protect_ratio


    def start_ros_handlers(self):
        """ after initializing the env and policy, register ros related callbacks and topics
        """

        # ROS publishers
        self.low_cmd_pub = self.create_publisher(
            LowCmd,
            self.low_cmd_topic,
            1
        )
        self.low_cmd_buffer = LowCmd()

        # ROS subscribers
        self.low_state_sub = self.create_subscription(
            LowState,
            self.low_state_topic,
            self._low_state_callback,
            1
        )

        self.sport_mode_state_sub = self.create_subscription(
            SportModeState,
            self.sport_mode_state_topic,
            self._sport_mode_state_callback,
            1
        )

        self.sport_state_pub = self.create_publisher(
            Request,
            '/api/robot_state/request',
            1,
        )

        self.sport_mode_pub = self.create_publisher(
            Request,
            '/api/sport/request',
            1,
        )

        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )

        if self.forward_depth_topic is not None:
            self.forward_camera_sub = self.create_subscription(
                Image,
                self.forward_depth_topic,
                self._forward_depth_callback,
                1
            )

        if self.forward_depth_embedding_topic is not None and "forward_depth" in self.replace_obs_with_embeddings:
            self.forward_depth_embedding_sub = self.create_subscription(
                Float32MultiArray,
                self.forward_depth_embedding_topic,
                self._forward_depth_embedding_callback,
                1,
            )

        self.get_logger().info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.get_logger().warn(f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep safe.")
        else:
            self.get_logger().warn(f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be safe.")
        while rclpy.ok():
            rclpy.spin_once(self)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.get_logger().info("Low state message received, the robot is ready to go.")

    """ ROS callbacks and handlers that update the buffer """
    def _low_state_callback(self, msg):
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state
        # print("self.low_state_buffer motor state[0]: ", self.low_state_buffer.motor_state[0].q)
        # print("self.low_state_buffer motor state[3]: ", self.low_state_buffer.motor_state[3].q)
        # print("self.low_state_buffer motor state[6]: ", self.low_state_buffer.motor_state[6].q)
        # print("self.low_state_buffer motor state[9]: ", self.low_state_buffer.motor_state[9].q)
        # print("*"*50)
            
        # self.low_state_buffer motor state[0]:  -0.049813926219940186
        # self.low_state_buffer motor state[3]:  0.05414363741874695
        # self.low_state_buffer motor state[6]:  -0.09193509817123413
        # self.low_state_buffer motor state[9]:  0.08324539661407471

        # refresh dof_pos and dof_vel
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_pos_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.dof_signs[sim_idx]

        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_vel_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.dof_signs[sim_idx]

        

        # automatic safety check
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if self.dof_pos_[0, sim_idx] > self.joint_pos_protect_high[sim_idx] or \
                self.dof_pos_[0, sim_idx] < self.joint_pos_protect_low[sim_idx]:
                self.get_logger().error(f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at {self.low_state_buffer.motor_state[real_idx].q}")
                self.get_logger().error(f"self.joint_pos_protect_low[sim_idx] is {self.joint_pos_protect_low[sim_idx]}")
                self.get_logger().error(f"self.joint_pos_protect_high[sim_idx] is {self.joint_pos_protect_high[sim_idx]}")
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
                raise SystemExit()
            
    def _sport_mode_state_callback(self, msg):
        """ store and handle proprioception data """
        self.sport_mode_state_buffer = msg # keep the latest sport mode state

    def _sport_state_change(self, mode):
        msg = Request()

        # Fill the header
        msg.header.identity.id = 0
        msg.header.identity.api_id = 1001
        msg.header.lease.id = 0
        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        # Fill the parameter
        if mode==0:
            msg.parameter = '{"name":"sport_mode","switch":0}'
        elif mode==1:
            msg.parameter = '{"name":"sport_mode","switch":1}'

        # Binary data (optional, leave empty if not needed)
        msg.binary = []

        # Publish the request
        self.sport_state_pub.publish(msg)
        # self.get_logger().info(f"Request sent: {msg}")


    def _sport_mode_change(self, mode):
        msg = Request()

        # Fill the request header for damp mode
        msg.header.identity.id = 0  # Replace with appropriate ID if required
        msg.header.identity.api_id = mode  # ID for damp mode
        msg.header.lease.id = 0  # Lease ID
        msg.header.policy.priority = 0  # Priority level
        msg.header.policy.noreply = False  # Expect a response

        # Parameter and binary data can remain empty for this mode
        msg.parameter = ''
        msg.binary = []

        # Publish the request
        self.sport_mode_pub.publish(msg)
        # self.get_logger().info(f"Request sent: {msg}")

    def _joy_stick_callback(self, msg):
        self.joy_stick_buffer = msg
        if self.move_by_wireless_remote:
            # left-y for forward/backward
            ly = msg.ly
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for turning left/right
            lx = -msg.lx
            if lx > self.ang_vel_deadband:
                yaw = (lx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif lx < -self.ang_vel_deadband:
                yaw = (lx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0
            # right-x for side moving left/right
            rx = -msg.rx
            if rx > self.lin_vel_deadband:
                vy = (rx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif rx < -self.lin_vel_deadband:
                vy = (rx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            self.xyyaw_command = torch.tensor([vx, vy, yaw], device= self.model_device, dtype= torch.float32)

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        # if (msg.keys & self.WirelessButtons.R2) or (msg.keys & self.WirelessButtons.L2): # R2 or L2 is pressed
        #     self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
        #     self._turn_off_motors()
        #     raise SystemExit()

        # roll-pitch target
        if hasattr(self, "roll_pitch_yaw_cmd"):
            if (msg.keys & self.WirelessButtons.up):
                self.roll_pitch_yaw_cmd[0, 1] += 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.down):
                self.roll_pitch_yaw_cmd[0, 1] -= 0.1
                self.get_logger().info("Pitch Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.left):
                self.roll_pitch_yaw_cmd[0, 0] -= 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))
            if (msg.keys & self.WirelessButtons.right):
                self.roll_pitch_yaw_cmd[0, 0] += 0.1
                self.get_logger().info("Roll Command: " + str(self.roll_pitch_yaw_cmd))


    def _forward_depth_callback(self, msg):
        """ store and handle depth camera data """
        normized_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.forward_depth_buffer = torch.tensor(normized_depth_image, dtype=torch.float32).unsqueeze(0)


    def _forward_depth_embedding_callback(self, msg):
        self.forward_depth_embedding_buffer = torch.tensor(msg.data, device= self.model_device, dtype= torch.float32).view(1, -1)

    """ Done: ROS callbacks and handlers that update the buffer """
    """ refresh observation buffer and corresponding sub-functions """


    def _get_ang_vel_obs(self):
        buffer = torch.from_numpy(self.low_state_buffer.imu_state.gyroscope).unsqueeze(0)
        return buffer

    def _get_commands_obs(self):
        return self.xyyaw_command.unsqueeze(0) # (1, 3)
    
    def _get_dof_vel_obs(self):
        return self.dof_vel_

    def _get_last_actions_obs(self):
        return self.actions

    def _get_depth_obs(self):
        return self.forward_depth_buffer
    
    def _get_dof_pos_obs(self):
        return self.reindex(self.dof_pos_ - self.default_dof_pos.unsqueeze(0))
    
    def _get_contact_filt_obs(self):
        contact = [force > 20 for force in self.low_state_buffer.foot_force]
        contact_filt = [contact[1], contact[0], contact[3], contact[2]]
        self.contact_filt = np.logical_or(contact_filt, self.last_contacts)
        self.last_contacts = contact_filt
        final_contact_vec = torch.tensor(self.contact_filt).float().unsqueeze(0).to(self.model_device)
        return final_contact_vec


    def reindex(self, vec: torch.Tensor):
        if vec is not None:
            if vec.shape != (1,12):
                print("Error: The shape of the reindex vec is not correct. Expected shape is (1, 12).")
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
    

    def reindex_feet(self, vec: torch.Tensor):
        if vec is not None:
            if vec.shape != (1,4):
                print("Error: The shape of the reindex_feet vec is not correct. Expected shape is (1, 4).")
        return vec[:, [1, 0, 3, 2]]
    

    def read_observation(self):
        self.step_count += 1

        ######################################################################################
        ############################## Record sensor data ####################################
        ######################################################################################
        # imu = self.low_state_buffer.imu_state

        # # Get contact data and reindex it
        # contact = self.reindex_feet(self._get_contact_filt_obs() - 0.5)

        # # Record data: step_count, 3 gyro, 3 rpy, 4 contacts
        # log_entry = [
        #     self.step_count,
        #     imu.gyroscope[0],
        #     imu.gyroscope[1],
        #     imu.gyroscope[2],
        #     imu.rpy[0],
        #     imu.rpy[1],
        #     imu.rpy[2],
        #     *contact.flatten().tolist()  # Flatten tensor to list
        # ]
        # self.data_log.append(log_entry)

        # if self.step_count % 20 == 0:
        #     save_path = os.path.expanduser("~/parkour/plot/full_sensor_log.npy")
        #     np.save(save_path, np.array(self.data_log)) # shape: (step, 11)
        #####################################################################################


        # Convert all placeholders to tensors
        placeholder_base_ang_vel = torch.tensor(
            [x * 0.25 for x in self.low_state_buffer.imu_state.gyroscope],
            device=self.model_device, dtype=torch.float32
        )   # [1,3]

        
        # placeholder_base_ang_vel[2] = placeholder_base_ang_vel[2] + (self.low_state_buffer.imu_state.rpy[2]) * 0.3
        # placeholder_base_ang_vel[2] = 0
        

        placeholder_imu_obs = torch.tensor(
            [self.low_state_buffer.imu_state.rpy[0], self.low_state_buffer.imu_state.rpy[1]],
            device=self.model_device, dtype=torch.float32
        )   # [1,2]

        placeholder_0_delta_yaw = torch.tensor([0], device=self.model_device, dtype=torch.float32)   # [1,1]
        placeholder_delta_yaw = torch.tensor([0], device=self.model_device, dtype=torch.float32)   # will be predicted by depth_encoder
        placeholder_delta_next_yaw = torch.tensor([0], device=self.model_device, dtype=torch.float32)  # will be predicted by depth_encoder
        placeholder_0_commands = torch.tensor([0, 0], device=self.model_device, dtype=torch.float32)  
        placeholder_commands = torch.tensor([0.5403], device=self.model_device, dtype=torch.float32)  # self.commands[:, 0]. it is a random velocity command for x direction, range [0.3, 0.8]
        placeholder_env_class_not_17 = torch.tensor([1], device=self.model_device, dtype=torch.float32)
        placeholder_env_class_17 = torch.tensor([0], device=self.model_device, dtype=torch.float32)

        placeholder_dof_pos = torch.tensor(
            self.reindex((self.dof_pos_ - self.default_dof_pos.unsqueeze(0)) * 1.0),
            device=self.model_device, dtype=torch.float32
        )
        # print("placeholder_dof_pos: ", placeholder_dof_pos)
        
        placeholder_dof_vel = torch.tensor(
            self.reindex(self.dof_vel_ * 0.05),
            device=self.model_device, dtype=torch.float32
        )
        placeholder_action_history_buf = torch.tensor(
            self.actions,
            device=self.model_device, dtype=torch.float32
        )
        placeholder_contact_filt = torch.tensor(
            self.reindex_feet(self._get_contact_filt_obs() - 0.5),
            device=self.model_device, dtype=torch.float32
        )

        # Concatenate placeholders to create `obs_buf`
        obs_buf = torch.cat([
            placeholder_base_ang_vel,
            placeholder_imu_obs,
            placeholder_0_delta_yaw,
            placeholder_delta_yaw,
            placeholder_delta_next_yaw,
            placeholder_0_commands,
            placeholder_commands,
            placeholder_env_class_not_17,
            placeholder_env_class_17,
            placeholder_dof_pos.flatten(),
            placeholder_dof_vel.flatten(),
            placeholder_action_history_buf.flatten(),
            placeholder_contact_filt.flatten()
        ])  # size 53

        # Convert other arrays to tensors
        velocity_tensor = torch.tensor(
            self.sport_mode_state_buffer.velocity * 2.0, 
            dtype=torch.float32, 
            device=self.model_device
        )
        zeros_tensor = torch.zeros(6, dtype=torch.float32, device=self.model_device)
        priv_explicit = torch.cat([velocity_tensor, zeros_tensor])

        priv_latent = torch.tensor(
            [0.0000,  0.0000,  0.0000,  0.0000,  1.5996,  0.0688, -0.1073,  0.0870,
            -0.0627,  0.1623,  0.0584, -0.1670,  0.0028,  0.0276,  0.1598,  0.1258,
            -0.1390, -0.1381,  0.1247, -0.0358, -0.1432,  0.1187, -0.1178, -0.0393,
            0.0180, -0.1365,  0.1777,  0.1946,  0.1676],
            device=self.model_device, dtype=torch.float32
        )
        heights = torch.zeros(132, device=self.model_device, dtype=torch.float32)

        # Handle history buffer
        if self.step_count <= 1:
            self.obs_history_buf = torch.cat([obs_buf] * 10)

        self.total_obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf])

        if self.step_count > 1:
            self.obs_history_buf = torch.cat((self.obs_history_buf[53:], obs_buf))

        # Clip the observations
        self.obs_buffer = (torch.clamp(self.total_obs_buf, -self.clip_obs, self.clip_obs)).unsqueeze(0)
        return self.obs_buffer

    def send_action(self, actions: torch.Tensor):
        if actions is not None:
            if actions.shape != (1,12):
                print("Error: The shape of actions is not correct. Expected shape is (1,12).")
            else:
                self.actions = actions

        actions = self.clip_action_before_scale(actions)
        # clipped_scaled_action = self.clip_by_torque_limit(actions * self.action_scale)

        robot_coordinates_action = actions * self.action_scale + self.reindex(self.default_dof_pos.unsqueeze(0))
        # print("self.reindex(self.default_dof_pos.unsqueeze(0)) is ", self.reindex(self.default_dof_pos.unsqueeze(0)))

        self._publish_legs_cmd(robot_coordinates_action[0])

    def clip_action_before_scale(self, actions):
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        return actions
    
    def clip_by_torque_limit(self, actions_scaled):
        """ Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """
        p_limits_low = (-self.torque_limits) + self.d_gains*self.dof_vel_
        p_limits_high = (self.torque_limits) + self.d_gains*self.dof_vel_
        actions_low = (p_limits_low/self.p_gains) - self.default_dof_pos + self.dof_pos_
        actions_high = (p_limits_high/self.p_gains) - self.default_dof_pos + self.dof_pos_
        return torch.clip(actions_scaled, actions_low, actions_high)   

    
    """ functions that actually publish the commands and take effect """
    def _publish_legs_cmd(self, robot_coordinates_action: torch.Tensor):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in simulation order.
        """
        # for sim_idx in range(self.NUM_DOF):
        #     real_idx = self.dof_map[sim_idx]
        #     if not self.dryrun:
        #         self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
        #     self.low_cmd_buffer.motor_cmd[real_idx].q = robot_coordinates_action[sim_idx].item() * self.dof_signs[sim_idx]
        #     self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
        #     self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
        #     self.low_cmd_buffer.motor_cmd[real_idx].kp = self.p_gains
        #     self.low_cmd_buffer.motor_cmd[real_idx].kd = self.d_gains

        for real_idx in range(self.NUM_DOF):
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[real_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = robot_coordinates_action[real_idx].item() * self.dof_signs[real_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.p_gains
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.d_gains

        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
    """ Done: functions that actually publish the commands and take effect """