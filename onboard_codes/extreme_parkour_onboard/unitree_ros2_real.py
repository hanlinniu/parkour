import os, sys

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    SportModeState,
    LowCmd,
)

from unitree_api.msg import Request, RequestHeader
from sensor_msgs.msg import Image, CameraInfo

from std_msgs.msg import Float32MultiArray
if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "x86",
    ))
elif os.uname().machine == "aarch64":
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "aarch64",
    ))
from crc_module import get_crc

import numpy as np
import torch
import time
from cv_bridge import CvBridge


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = 1.0 - 2.0 * (q[:, qx] * q[:, qx] + q[:, qy] * q[:, qy])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = 1.0 - 2.0 * (q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class RobotCfgs:
    class H1:
        pass

    class Go2:
        NUM_DOF = 12
        NUM_ACTIONS = 12
        # The order of joints has been reindexed in simulation.
        # So we do not need here.
        dof_map = [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
        ]
        dof_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        dof_signs = [1.] * 12
        joint_limits_high = torch.tensor([
            1.0472, 3.4907, -0.83776,
            1.0472, 3.4907, -0.83776,
            1.0472, 4.5379, -0.83776,
            1.0472, 4.5379, -0.83776,
        ], device= "cuda", dtype= torch.float32)
        joint_limits_low = torch.tensor([
            -1.0472, -1.5708, -2.7227,
            -1.0472, -1.5708, -2.7227,
            -1.0472, -0.5236, -2.7227,
            -1.0472, -0.5236, -2.7227,
        ], device= "cuda", dtype= torch.float32)
        torque_limits = torch.tensor([ # from urdf
            25, 40, 40,
            25, 40, 40,
            25, 40, 40,
            25, 40, 40,
        ], device= "cuda", dtype= torch.float32)
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
            sport_mode_state_topic = "/sportmodestate",
            low_cmd_topic= "/lowcmd",
            joy_stick_topic= "/wirelesscontroller",
            forward_depth_topic= "/camera/forward_depth", # if None and still need access, set to str "pyrealsense"
            depth_data_topic= "/forward_depth_image",
            cfg= dict(),
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            cmd_px_range= [0.4, 1.0], # check joy_stick_callback (p for positive, n for negative)
            cmd_nx_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_py_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_ny_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_pyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            cmd_nyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
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
        self.sport_mode_state_topic = sport_mode_state_topic
        self.low_cmd_topic = low_cmd_topic if not dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        self.joy_stick_topic = joy_stick_topic
        self.forward_depth_topic = forward_depth_topic
        self.depth_data_topic = depth_data_topic
        self.cfg = cfg
        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.cmd_px_range = cmd_px_range
        self.cmd_nx_range = cmd_nx_range
        self.cmd_py_range = cmd_py_range
        self.cmd_ny_range = cmd_ny_range
        self.cmd_pyaw_range = cmd_pyaw_range
        self.cmd_nyaw_range = cmd_nyaw_range
        self.move_by_wireless_remote = move_by_wireless_remote
        self.model_device = model_device
        self.dof_pos_protect_ratio = dof_pos_protect_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun

        self.bridge = CvBridge()


        self.dof_map = getattr(RobotCfgs, robot_class_name).dof_map
        self.dof_names = getattr(RobotCfgs, robot_class_name).dof_names
        self.dof_signs = getattr(RobotCfgs, robot_class_name).dof_signs
        self.turn_on_motor_mode = getattr(RobotCfgs, robot_class_name).turn_on_motor_mode

        # self.n_obs = 
        self.n_proprio = 53
        self.n_depth_latent = 32
        self.n_hist_len = 10

        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.model_device, dtype=torch.float)
        self.episode_length_buf = torch.zeros(1, device=self.model_device, dtype=torch.float)
        self.xyyaw_command = torch.tensor([[0, 0, 0]], device= self.model_device, dtype= torch.float32)
        self.contact_filt = torch.zeros((1, 4), device= self.model_device, dtype= torch.float32)
        # self.last_contact_filt = torch.zeros((1, 4), device= self.model_device, dtype= torch.float32)

        self.parse_config()
        

    def parse_config(self):
        """ parse, set attributes from config dict, initialize buffers to speed up the computation """

        # observation
        self.clip_obs = self.cfg["normalization"]["clip_observations"]

        # controls
        self.control_type = self.cfg["control"]["control_type"]
        if not (self.control_type == "P"):
            raise NotImplementedError("Only position control is supported for now.")
        
        self.p_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            for k, v in self.cfg["control"]["stiffness"].items():
                if k in name:
                    self.p_gains.append(v)
                    break 
        self.p_gains = torch.tensor(self.p_gains, device= self.model_device, dtype= torch.float32)

        self.d_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i] 
            for k, v in self.cfg["control"]["damping"].items():
                if k in name:
                    self.d_gains.append(v)
                    break
        self.d_gains = torch.tensor(self.d_gains, device= self.model_device, dtype= torch.float32)

        self.default_dof_pos = torch.zeros(self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_pos_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        self.dof_vel_ = torch.empty(1, self.NUM_DOF, device= self.model_device, dtype= torch.float32)
        
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            default_joint_angle = self.cfg["init_state"]["default_joint_angles"][name]
            self.default_dof_pos[i] = default_joint_angle

        # actions
        self.num_actions = self.NUM_ACTIONS
        self.action_scale = self.cfg["control"]["action_scale"]
        self.get_logger().info("[Env] action scale: {:.2f}".format(self.action_scale))
        self.clip_actions = self.cfg["normalization"]["clip_actions"]
        if self.cfg["normalization"].get("clip_actions_method", None) == "hard":
            self.get_logger().info("clip_actions_method with hard mode")
            self.get_logger().info("clip_actions_high: " + str(self.cfg["normalization"]["clip_actions_high"]))
            self.get_logger().info("clip_actions_low: " + str(self.cfg["normalization"]["clip_actions_low"]))
            self.clip_actions_method = "hard"
            self.clip_actions_low = torch.tensor(self.cfg["normalization"]["clip_actions_low"], device= self.model_device, dtype= torch.float32)
            self.clip_actions_high = torch.tensor(self.cfg["normalization"]["clip_actions_high"], device= self.model_device, dtype= torch.float32)
        else:
            self.get_logger().info("clip_actions_method is " + str(self.cfg["normalization"].get("clip_actions_method", None)))
        
        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)    

        ###################### hardware related #####################
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
        self.get_logger().info("Low state subscriber started, waiting to receive low state messages.")

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
        self.get_logger().info("Wireless controller subscriber started, waiting to receive wireless controller messages.")

        if self.forward_depth_topic is not None:
            self.forward_camera_sub = self.create_subscription(
                Image,
                self.forward_depth_topic,
                self._forward_depth_callback,
                1
            )


        # self.depth_input_sub = self.create_subscription(
        #     Float32MultiArray,
        #     self.depth_data_topic,
        #     self._depth_data_callback,
        #     1
        # )


        self.get_logger().info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.get_logger().warn(f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep safe.")
        else:
            self.get_logger().warn(f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be safe.")
        while rclpy.ok():
            rclpy.spin_once(self)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.get_logger().info("Low state and wireless message received, the robot is ready to go.")

    """ ROS callbacks and handlers that update the buffer """

    def _low_state_callback(self, msg):
        # self.get_logger().info("Low state message received.")
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state

        ################### refresh dof_pos and dof_vel ######################
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
    def _joy_stick_callback(self, msg):
        # self.get_logger().info("Wireless controller message received.")
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
            self.xyyaw_command = torch.tensor([[0.5, vy, yaw]], device= self.model_device, dtype= torch.float32)

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        # if (msg.keys & self.WirelessButtons.R2) or (msg.keys & self.WirelessButtons.L2): # R2 or L2 is pressed
        #     self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
        #     self._turn_off_motors()
        #     raise SystemExit()

        # roll-pitch target, not implemented yet
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
        # self.forward_depth_buffer = torch.tensor(normized_depth_image, dtype=torch.float32, device=self.model_device).unsqueeze(0)
        self.forward_depth_buffer = torch.from_numpy(normized_depth_image).float().unsqueeze(0).to(self.model_device)


    def _depth_data_callback(self, msg):
        self.depth_data = torch.tensor(msg.data, dtype=torch.float32).reshape(1, 58, 87).to(self.model_device)

    """ Done: ROS callbacks and handlers that update the buffer """

    """ refresh observation buffer and corresponding sub-functions """
    
    def _get_ang_vel_obs(self):
        ang_vel = torch.from_numpy(self.low_state_buffer.imu_state.gyroscope).unsqueeze(0).to(device=self.model_device, dtype=torch.float32)
        return ang_vel * self.cfg["normalization"]["obs_scales"]["ang_vel"]
    
    def _get_imu_obs(self):
        quat_xyzw = torch.tensor([
            self.low_state_buffer.imu_state.quaternion[1],
            self.low_state_buffer.imu_state.quaternion[2],
            self.low_state_buffer.imu_state.quaternion[3],
            self.low_state_buffer.imu_state.quaternion[0],
            ], device= self.model_device, dtype= torch.float32).unsqueeze(0)
        roll, pitch, yaw = get_euler_xyz(quat_xyzw)
        imu_obs = torch.tensor([[roll, pitch]], device= self.model_device, dtype= torch.float32)
        return imu_obs

    def _get_delta_yaw_obs(self):
        yaw = 0
        delta_yaw, delta_next_yaw = 0, 0
        yaw_info = torch.tensor([[0, delta_yaw, delta_next_yaw]], device= self.model_device, dtype= torch.float32)
        return yaw_info

    #  maybe only vx used
    def _get_commands_obs(self):
        if self.move_by_wireless_remote:
            vx, _, _ = self.xyyaw_command[0, :]
            commands = torch.tensor([[0, 0, vx]], device= self.model_device, dtype= torch.float32)
            return commands
        else:
            return torch.tensor([[0., 0., 0.]], device=self.model_device)

    def _get_dof_pos_obs(self):
        return (self.dof_pos_ - self.default_dof_pos.unsqueeze(0)) * self.cfg["normalization"]["obs_scales"]["dof_pos"]

    def _get_dof_vel_obs(self):
        return self.dof_vel_ * self.cfg["normalization"]["obs_scales"]["dof_vel"]

    def _get_last_actions_obs(self):
        return self.actions

    def _get_contact_filt_obs(self):
        for i in range(4):
            if self.low_state_buffer.foot_force[i] < 25:
                self.contact_filt[:, i] = -0.5
            else:
                self.contact_filt[:, i] = 0.5
        return self.contact_filt
        # self.contact = torch.tensor(self.low_state_buffer.foot_force, dtype=torch.float32).unsqueeze(0).to(device=self.model_device) >  25.
        # print('contact: ', self.contact)
        # self.contact_filt = torch.logical_or(self.contact, self.last_contact_filt).float()
        # print('contact filt: ', self.contact_filt)
        # print('last contact filt: ', self.last_contact_filt)
        # self.last_contact_filt = self.contact
        # return self.contact_filt - 0.5

    def _get_depth_obs(self):
        return self.forward_depth_buffer
    
    def _get_depth_image(self):
        return self.depth_data

    
    def _get_history_proprio(self):
        return self.proprio_history_buf
    

    def get_proprio(self):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        start_time = time.monotonic()

        ang_vel = self._get_ang_vel_obs()  # (1, 3)
        ang_vel_time = time.monotonic()

        imu = self._get_imu_obs()  # (1, 2)
        imu_time = time.monotonic()

        yaw_info = self._get_delta_yaw_obs()  # (1, 3)
        yaw_time = time.monotonic()

        commands = self._get_commands_obs()  # (1, 3)
        commands_time = time.monotonic()

        parkour_walk = torch.tensor([[1, 0]], device= self.model_device, dtype= torch.float32) # parkour
        # parkour_walk = torch.tensor([[0, 1]], device= self.model_device, dtype= torch.float32) # walk

        dof_pos = self._get_dof_pos_obs()  # (1, 12)
        dof_pos_time = time.monotonic()

        dof_vel = self._get_dof_vel_obs()  # (1, 12)
        dof_vel_time = time.monotonic()

        last_actions = self._get_last_actions_obs().view(1, -1)  # (1, 12)
        last_action_time = time.monotonic()

        contact = self._get_contact_filt_obs()  # (1, 4)
        contact_time = time.monotonic()
        
        proprio = torch.cat([ang_vel, imu, yaw_info, commands, parkour_walk,
                        dof_pos, dof_vel,
                        last_actions, 
                        contact], dim=-1)

        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([proprio] * self.n_hist_len, dim=1),
            torch.cat([
                self.proprio_history_buf[:, 1:],
                proprio.unsqueeze(1)
            ], dim=1)
        )
        end_time = time.monotonic()

        # print('ang vel time: {:.5f}'.format(ang_vel_time - start_time),
        #         'imu time: {:.5f}'.format(imu_time - ang_vel_time),
        #         'yaw time: {:.5f}'.format(yaw_time - imu_time),
        #         'command time: {:.5f}'.format(commands_time - yaw_time),
        #         'dof pos time: {:.5f}'.format(dof_pos_time - commands_time),
        #         'dof vel time: {:.5f}'.format(dof_vel_time - dof_pos_time),
        #         'last action time: {:.5f}'.format(last_action_time - dof_vel_time),
        #         'contact time: {:.5f}'.format(contact_time - last_action_time)
        #         )
        
        self.episode_length_buf += 1

        return proprio



    """ Done: refresh observation buffer and corresponding sub-functions """

    """ Control related functions """
    def clip_action_before_scale(self, action):
        action = torch.clip(action, -self.clip_actions, self.clip_actions)
        return action

    def clip_by_torque_limit(self, actions_scaled):
        control_type = self.cfg["control"]["control_type"]
        if control_type == "P":
            p_limits_low = (-self.torque_limits) + self.d_gains*self.dof_vel_
            p_limits_high = (self.torque_limits) + self.d_gains*self.dof_vel_
            actions_low = (p_limits_low/self.p_gains) - self.default_dof_pos + self.dof_pos_
            actions_high = (p_limits_high/self.p_gains) - self.default_dof_pos + self.dof_pos_
            print('action low and high: ', actions_low, actions_high)
        else:
            raise NotImplementedError

        return torch.clip(actions_scaled, actions_low, actions_high)

    def reset_obs_buffers(self):
        self.step_count = 0
        self.obs_buffer = torch.empty(0, device=self.model_device)
        self.obs_history_buf = torch.empty(0, device=self.model_device)
        self.total_obs_buf = torch.empty(0, device=self.model_device)
        self.actions = torch.zeros(self.NUM_ACTIONS, device= self.model_device, dtype= torch.float32)
        self.episode_length_buf = torch.zeros(1, device=self.model_device, dtype=torch.float)
        self.get_logger().info("Observation buffers have been reset!")
        
    def send_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        self.actions = actions

        hard_clip = self.cfg["normalization"]["clip_actions"]/self.cfg["control"]["action_scale"]
        clipped_scaled_action = torch.clip(actions, -hard_clip, hard_clip) * self.cfg["control"]["action_scale"]
        
        robot_coordinates_action = clipped_scaled_action + self.default_dof_pos.unsqueeze(0)
        # print('coordinates action: ', robot_coordinates_action)
        self._publish_legs_cmd(robot_coordinates_action[0])


    """ functions that actually publish the commands and take effect """
    def _publish_legs_cmd(self, robot_coordinates_action: torch.Tensor):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in simulation order.
        """
        #################### check ##############################
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = robot_coordinates_action[sim_idx].item() * self.dof_signs[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.p_gains[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.d_gains[sim_idx].item()
        
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
