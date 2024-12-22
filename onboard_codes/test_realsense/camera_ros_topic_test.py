import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo

import os
import os.path as osp
import json
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


import numpy as np
import cv2


import pyrealsense2 as rs
import ros2_numpy as rnp

@torch.no_grad()
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data

class VisualHandlerNode(Node):
    """ A wapper class for the realsense camera """
    def __init__(self,
            cfg: dict,
            cropping: list = [0, 0, 0, 0], # top, bottom, left, right
            rs_resolution: tuple = (480, 270), # width, height for the realsense camera)
            rs_fps: int= 30,
            depth_input_topic= "/camera/forward_depth",
            rgb_topic= "/camera/forward_rgb",
            camera_info_topic= "/camera/camera_info",
            enable_rgb= False,
            forward_depth_embedding_topic= "/forward_depth_embedding",
        ):
        # super().__init__("forward_depth_embedding")
        super().__init__("visual_handler_node")
        self.cfg = cfg
        self.cropping = cropping
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.depth_input_topic = depth_input_topic
        self.rgb_topic= rgb_topic
        self.camera_info_topic = camera_info_topic
        self.enable_rgb= enable_rgb
        self.forward_depth_embedding_topic = forward_depth_embedding_topic

        self.parse_args()
        self.start_pipeline()
        self.start_ros_handlers()

    def parse_args(self):
        self.output_resolution = self.cfg["sensor"]["forward_camera"].get(
            "output_resolution",
            self.cfg["sensor"]["forward_camera"]["resolution"],
        )
        depth_range = self.cfg["sensor"]["forward_camera"].get(
            "depth_range",
            [0.0, 3.0],
        )
        self.depth_range = (depth_range[0] * 1000, depth_range[1] * 1000) # [m] -> [mm]

    def start_pipeline(self):
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.depth,
            self.rs_resolution[0],
            self.rs_resolution[1],
            rs.format.z16,
            self.rs_fps,
        )
        if self.enable_rgb:
            self.rs_config.enable_stream(
                rs.stream.color,
                self.rs_resolution[0],
                self.rs_resolution[1],
                rs.format.rgb8,
                self.rs_fps,
            )
        self.rs_profile = self.rs_pipeline.start(self.rs_config)

        self.rs_align = rs.align(rs.stream.depth)

        # build rs builtin filters
        # self.rs_decimation_filter = rs.decimation_filter()
        # self.rs_decimation_filter.set_option(rs.option.filter_magnitude, 6)
        self.rs_hole_filling_filter = rs.hole_filling_filter()
        self.rs_spatial_filter = rs.spatial_filter()
        self.rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_spatial_filter.set_option(rs.option.holes_fill, 4)
        self.rs_temporal_filter = rs.temporal_filter()
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)
        # using a list of filters to define the filtering order
        self.rs_filters = [
            # self.rs_decimation_filter,
            self.rs_hole_filling_filter,
            self.rs_spatial_filter,
            self.rs_temporal_filter,
        ]

        if self.enable_rgb:
            # get frame with longer waiting time to start the system
            # I know what's going on, but when enabling rgb, this solves the problem.
            rs_frame = self.rs_pipeline.wait_for_frames(int(
                self.cfg["sensor"]["forward_camera"]["latency_range"][1] * 10000 # ms * 10
            ))

    def start_ros_handlers(self):
        self.depth_input_pub = self.create_publisher(
            Image,
            self.depth_input_topic,
            1,
        )
        if self.enable_rgb:
            self.rgb_pub = self.create_publisher(
                Image,
                self.rgb_topic,
                1,
            )
            self.camera_info_pub = self.create_publisher(
                CameraInfo,
                self.camera_info_topic,
                1,
            )
            # fill in critical info of processed camera info based on simulated data
            # NOTE: simply because realsense's camera_info does not match our network input.
            # It is easier to compute this way.
            self.camera_info_msg = CameraInfo()
            self.camera_info_msg.header.frame_id = "d435_sim_depth_link"
            self.camera_info_msg.height = self.output_resolution[0]
            self.camera_info_msg.width = self.output_resolution[1]
            self.camera_info_msg.distortion_model = "plumb_bob"
            self.camera_info_msg.d = [0., 0., 0., 0., 0.]
            sim_raw_resolution = self.cfg["sensor"]["forward_camera"]["resolution"]
            sim_cropping_h = self.cfg["sensor"]["forward_camera"]["crop_top_bottom"]
            sim_cropping_w = self.cfg["sensor"]["forward_camera"]["crop_left_right"]
            cropped_resolution = [ # (H, W)
                sim_raw_resolution[0] - sum(sim_cropping_h),
                sim_raw_resolution[1] - sum(sim_cropping_w),
            ]
            network_input_resolution = self.cfg["sensor"]["forward_camera"]["output_resolution"]
            x_fov = sum(self.cfg["sensor"]["forward_camera"]["horizontal_fov"]) / 2 / 180 * np.pi
            fx = (sim_raw_resolution[1]) / (2 * np.tan(x_fov / 2))
            fy = fx
            fx = fx * network_input_resolution[1] / cropped_resolution[1]
            fy = fy * network_input_resolution[0] / cropped_resolution[0]
            cx = (sim_raw_resolution[1] / 2) - sim_cropping_w[0]
            cy = (sim_raw_resolution[0] / 2) - sim_cropping_h[0]
            cx = cx * network_input_resolution[1] / cropped_resolution[1]
            cy = cy * network_input_resolution[0] / cropped_resolution[0]
            self.camera_info_msg.k = [
                fx, 0., cx,
                0., fy, cy,
                0., 0., 1.,
            ]
            self.camera_info_msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
            self.camera_info_msg.p = [
                fx, 0., cx, 0.,
                0., fy, cy, 0.,
                0., 0., 1., 0.,
            ]
            self.camera_info_msg.binning_x = 0
            self.camera_info_msg.binning_y = 0
            self.camera_info_msg.roi.do_rectify = False
            self.create_timer(
                self.cfg["sensor"]["forward_camera"]["refresh_duration"],
                self.publish_camera_info_callback,
            )

        self.forward_depth_embedding_pub = self.create_publisher(
            Float32MultiArray,
            self.forward_depth_embedding_topic,
            1,
        )
        self.get_logger().info("ros handlers started")

    def publish_camera_info_callback(self):
        self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info("camera info published", once= True)
        self.camera_info_pub.publish(self.camera_info_msg)

    def get_depth_frame(self):
        # read from pyrealsense2, preprocess and write the model embedding to the buffer
        rs_frame = self.rs_pipeline.wait_for_frames(int(
            self.cfg["sensor"]["forward_camera"]["latency_range"][1] * 1000 # ms
        ))

        depth_frame = rs_frame.get_depth_frame()
        if not depth_frame:
            self.get_logger().error("No depth frame", throttle_duration_sec= 1)
            return
        
        # apply relsense filters
        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)


        # # this is process depth image using pytorch
        # depth_image_np = np.asanyarray(depth_frame.get_data())
        # depth_image_pyt = torch.from_numpy(depth_image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)   # the default shape is [480, 640]

        # depth_image_pyt = resize2d(depth_image_pyt, [60, 106])  # downsample
        # depth_image_pyt = depth_image_pyt[:, :, :-2, 4:-4]   # cropping

        # # # depth_image_pyt = - depth_image_pyt

        # depth_image_pyt = torch.clip(
        #     depth_image_pyt,
        #     self.cfg["sensor"]["forward_camera"]['near_clip'],
        #     self.cfg["sensor"]["forward_camera"]['far_clip'],
        # )
        # depth_input_data = (depth_image_pyt.detach().cpu().numpy()).astype(np.uint16)[0, 0] # (h, w) unit [mm]   # the rescaled depth values are implicitly converted from meters (float32) to millimeters (uint16). 



        ########################################################################################################################
        # this is process depth_image using numpy
        # Step 1: Get depth data as a NumPy array
        depth_image_np = np.asanyarray(depth_frame.get_data()).astype(np.uint16)  # shape [480, 640]

        # Step 2: Downsample using nearest neighbor interpolation
        depth_image_np_resized = cv2.resize(
            depth_image_np,
            (106, 60),  # target width and height
            interpolation=cv2.INTER_LINEAR
        )

        # Step 3: Crop the resized image
        depth_image_np_cropped = depth_image_np_resized[:-2, 4:-4]  # crop 2 pixels from bottom, 4 from both sides

        # Step 4: Clip the depth values
        near_clip = self.cfg["sensor"]["forward_camera"]['near_clip']
        far_clip = self.cfg["sensor"]["forward_camera"]['far_clip']
        depth_image_np_clipped = np.clip(depth_image_np_cropped, near_clip, far_clip)   # output shape is (58, 98)

        # Step 5: Convert to uint16 (scale to millimeters if necessary)
        # depth_input_data = (depth_image_np_clipped).astype(np.uint16)  # convert meters to millimeters
        ########################################################################################################################


        depth_input_msg = rnp.msgify(Image, depth_image_np_clipped, encoding= "16UC1")
        depth_input_msg.header.stamp = self.get_clock().now().to_msg()
        depth_input_msg.header.frame_id = "d435_sim_depth_link"
        self.depth_input_pub.publish(depth_input_msg)
        self.get_logger().info("depth input published", once= True)

        return depth_image_np_clipped
    

    def start_main_loop_timer(self, duration):
        self.create_timer(
            duration,
            self.main_loop,
        )

    def main_loop(self):
        depth_image_pyt = self.get_depth_frame()



def main(args):
    rclpy.init()

    camera_cfg = {
        "sensor": {
            "forward_camera": {
                "obs_components": ["forward_depth"],
                "resolution": [int(480 / 4), int(640 / 4)],
                "position": {
                    "mean": [0.24, -0.0175, 0.12],
                    "std": [0.01, 0.0025, 0.03],
                },
                "rotation": {
                    "lower": [-0.1, 0.37, -0.1],
                    "upper": [0.1, 0.43, 0.1],
                },
                "resized_resolution": [48, 64],
                "output_resolution": [48, 64],
                "horizontal_fov": [86, 90],
                "crop_top_bottom": [int(48 / 4), 0],
                "crop_left_right": [int(28 / 4), int(36 / 4)],
                "near_plane": 0.05,
                "depth_range": [0.0, 3.0],
                "latency_range": [0.08, 0.142],
                "latency_resample_time": 5.0,
                "refresh_duration": 1 / 10,  # [s]
                "far_clip": 2000,
                "near_clip": 0,
            }
        }
    }

    # assert args.logdir is not None, "Please provide a logdir"
    # with open(osp.join(args.logdir, "config.json"), "r") as f:
    #     config_dict = json.load(f, object_pairs_hook= OrderedDict)
        
    device = "cpu"
    duration = camera_cfg["sensor"]["forward_camera"]["refresh_duration"] # in sec

    visual_node = VisualHandlerNode(
        cfg= camera_cfg,
        cropping= [args.crop_top, args.crop_bottom, args.crop_left, args.crop_right],
        rs_resolution= (args.width, args.height),
        rs_fps= args.fps,
        enable_rgb= args.rgb,
    )

    visual_node.get_logger().info("Embedding send duration: {:.2f} sec".format(duration))
    visual_node.start_main_loop_timer(duration)
    rclpy.spin(visual_node)

    visual_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    
    parser.add_argument("--height",
        type= int,
        default= 480,
        help= "The height of the realsense image",
    )
    parser.add_argument("--width",
        type= int,
        default= 640,
        help= "The width of the realsense image",
    )
    parser.add_argument("--fps",
        type= int,
        default= 30,
        help= "The fps request to the rs pipeline",
    )
    parser.add_argument("--crop_left",
        type= int,
        default= 28,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_right",
        type= int,
        default= 36,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_top",
        type= int,
        default= 48,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_bottom",
        type= int,
        default= 0,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--rgb",
        action= "store_true",
        default= False,
        help= "Set to enable rgb visualization",
    )
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )

    args = parser.parse_args()
    main(args)

