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
from cv_bridge import CvBridge


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
            rs_resolution: tuple = (424, 240), # width, height for the realsense camera)   424, 240
            rs_fps: int= 90,
            depth_input_topic= "/camera/forward_depth",
            camera_info_topic= "/camera/camera_info",
            forward_depth_embedding_topic= "/forward_depth_embedding",
        ):
        # super().__init__("forward_depth_embedding")
        super().__init__("visual_handler_node")
        self.cfg = cfg
        self.cropping = cropping
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.depth_input_topic = depth_input_topic
        self.camera_info_topic = camera_info_topic
        self.forward_depth_embedding_topic = forward_depth_embedding_topic

        self.bridge = CvBridge()
        self.near_clip = 0
        self.far_clip = 2


        self.parse_args()
        self.start_pipeline()
        self.start_ros_handlers()

    def parse_args(self):
        pass
        # self.output_resolution = self.cfg["sensor"]["forward_camera"].get(
        #     "output_resolution",
        #     self.cfg["sensor"]["forward_camera"]["resolution"],
        # )
        # depth_range = self.cfg["sensor"]["forward_camera"].get(
        #     "depth_range",
        #     [0.0, 3.0],
        # )
        # self.depth_range = (depth_range[0] * 1000, depth_range[1] * 1000) # [m] -> [mm]

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

    def start_ros_handlers(self):
        self.depth_input_pub = self.create_publisher(
            Image,
            self.depth_input_topic,
            3,
        )

        self.forward_depth_embedding_pub = self.create_publisher(
            Float32MultiArray,
            self.forward_depth_embedding_topic,
            1,
        )
        self.get_logger().info("ros handlers started")


    def get_depth_frame(self):
        # read from pyrealsense2, preprocess and write the model embedding to the buffer
        # rs_frame = self.rs_pipeline.wait_for_frames(int(
        #     self.cfg["sensor"]["forward_camera"]["latency_range"][1] * 1000 # ms
        # ))

        rs_frame = self.rs_pipeline.wait_for_frames()
        depth_frame = rs_frame.get_depth_frame()
        if not depth_frame:
            self.get_logger().error("No depth frame", throttle_duration_sec= 1)
            return
        
        # apply relsense filters
        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)


        ########################################################################################################################
        # this is process depth_image using numpy
        # Step 1: Get depth data as a NumPy array
        # depth_image_np = np.asanyarray(depth_frame.get_data()).astype(np.uint16)  # shape [240, 424]
        depth_image_np = np.asanyarray(depth_frame.get_data()) / 1000.0  # Convert to meters  # shape [240, 424]
        

        # Step 2: Downsample using nearest neighbor interpolation
        depth_image_np_resized = cv2.resize(
            depth_image_np,
            (106, 60),  # target width and height
            interpolation=cv2.INTER_CUBIC
        )

        # Step 3: Crop the resized image
        depth_image_np_cropped = depth_image_np_resized[:-2, 4:-4]  # crop 2 pixels from bottom, 4 from both sides
        # print("original depth_image_np_cropped is ", depth_image_np_cropped)
        

        # Step 4: Clip the depth values
        depth_image_np_clipped = np.clip(depth_image_np_cropped, self.near_clip, self.far_clip)   # output shape is (58, 98)
        # print("depth_image_np_clipped is ", depth_image_np_clipped)

        # Step 5: Resize the depth image
        resized_depth_image_np_clipped = cv2.resize(depth_image_np_clipped, (87, 58), interpolation=cv2.INTER_CUBIC)  # output shape is (58, 87)

        # Step 6: Normize the depth image
        normized_depth_image = (resized_depth_image_np_clipped - self.near_clip) / (self.far_clip - self.near_clip) - 0.5
        # normized_depth_image = resized_depth_image_np_clipped

        # print("normized_depth_image shape is ", normized_depth_image.shape)
        # print("normized_depth_image is ", normized_depth_image)

        # Convert NumPy array to ROS Image message
        image_msg = self.bridge.cv2_to_imgmsg(normized_depth_image.astype(np.float32), encoding='32FC1')

        self.depth_input_pub.publish(image_msg)

        


        # Step 6: Convert to uint16 (scale to millimeters if necessary)
        # depth_input_data = (depth_image_np_clipped).astype(np.uint16)  # convert meters to millimeters
        ########################################################################################################################


        # depth_input_msg = rnp.msgify(Image, normized_depth_image, encoding= "16UC1")
        # depth_input_msg.header.stamp = self.get_clock().now().to_msg()
        # depth_input_msg.header.frame_id = "d435_sim_depth_link"
        # self.depth_input_pub.publish(depth_input_msg)
        # self.get_logger().info("depth input published", once= True)

        return normized_depth_image 
    

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
                "position": {
                    "mean": [0.24, -0.0175, 0.12],
                    "std": [0.01, 0.0025, 0.03],
                },
                "rotation": {
                    "lower": [-0.1, 0.37, -0.1],
                    "upper": [0.1, 0.43, 0.1],
                },
                "horizontal_fov": [86, 90],
                "crop_top_bottom": [int(48 / 4), 0],
                "crop_left_right": [int(28 / 4), int(36 / 4)],
                "near_plane": 0.05,
                "depth_range": [0.0, 2.0],
                "latency_range": [0.08, 0.142],
                "latency_resample_time": 5.0,
                "refresh_duration": 1 / 10,  # [s]
                "far_clip": 2,
                "near_clip": 0,
            }
        }
    }

    # assert args.logdir is not None, "Please provide a logdir"
    # with open(osp.join(args.logdir, "config.json"), "r") as f:
    #     config_dict = json.load(f, object_pairs_hook= OrderedDict)
        
    device = "cuda"
    duration = camera_cfg["sensor"]["forward_camera"]["refresh_duration"] # in sec

    visual_node = VisualHandlerNode(
        cfg= camera_cfg
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
    
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )

    args = parser.parse_args()
    main(args)

