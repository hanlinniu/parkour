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

import math
import pyrealsense2 as rs
import ros2_numpy as rnp

from torch_scatter import scatter_min


@torch.no_grad()
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data


def quat_apply_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q.
    q: [..., 4] (x, y, z, w)
    v: [..., 3]
    """
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]

    # Explicitly expand q_xyz and q_w if needed
    if q_xyz.shape[:-1] != v.shape[:-1]:
        q_xyz = q_xyz.expand_as(v)
        q_w = q_w.expand(*v.shape[:-1], 1)

    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (q_w * uv + uuv)


def quat_rotate_inverse_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by the inverse of quaternion q.
    q: [..., 4] in (x, y, z, w)
    v: [..., 3]
    """
    q_inv = torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)  # conjugate (unit quats)
    return quat_apply_batch(q_inv, v)  # your broadcast-aware apply

class VisualHandlerNode(Node):
    """ A wapper class for the realsense camera """
    def __init__(self,
            cfg: dict,
            cropping: list = [0, 0, 0, 0], # top, bottom, left, right
            rs_resolution: tuple = (424, 240), # width, height for the realsense camera)   424, 240
            rs_fps: int= 90,
            depth_input_topic= "/camera/forward_depth",
            depth_bev_input_topic = "/camera/forward_bev_depth",
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
        self.depth_bev_input_topic = depth_bev_input_topic
        self.camera_info_topic = camera_info_topic
        self.forward_depth_embedding_topic = forward_depth_embedding_topic

        self.bridge = CvBridge()
        self.near_clip = 0
        self.far_clip = 2

        self.episode_length_buf = 0

        self.device = torch.device("cuda")

        # Translation: (x_forward, y_left, z_up)
        self.camera_offsets = torch.tensor(
            [[0.3550, 0.0000, 0.0650]],
            dtype=torch.float32,
            device=self.device
        )

        # Quaternion: (x, y, z, w)
        self.camera_quats_offsets = torch.tensor(
            [[0.0000, 0.1981, 0.0000, 0.9802]],
            dtype=torch.float32,
            device=self.device
        )


        self.parse_args()
        self.start_pipeline()
        self.start_ros_handlers()

        rs_intrinsic_matrix = [55.79, 0.0, 53.0, 0.0, 54.15, 30.0, 0.0, 0.0, 1.0]
        self.rs_intrinsic_matrix = torch.tensor(rs_intrinsic_matrix, device=self.device).reshape(3, 3).unsqueeze(0)

        self.ray_starts, self.ray_directions = self._init_camera_rays(
            width=28,
            height=36, 
            intrinsic_matrices=self.rs_intrinsic_matrix, 
            device=self.device
            )
        
    def _init_camera_rays(self, width, height, intrinsic_matrices: torch.Tensor, device: str):
        # get image plane mesh grid
        grid = torch.meshgrid(
            torch.arange(start=0, end=width, dtype=torch.int32, device=device),
            torch.arange(start=0, end=height, dtype=torch.int32, device=device),
            indexing="xy",
        )
        pixels = torch.vstack(list(map(torch.ravel, grid))).T
        # convert to homogeneous coordinate system
        pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=device)])
        # move each pixel coordinate to the center of the pixel
        pixels += torch.tensor([[0.5, 0.5, 0]], device=device)
        # get pixel coordinates in camera frame
        pix_in_cam_frame = torch.matmul(torch.inverse(intrinsic_matrices), pixels.T)

        # robotics camera frame is (x forward, y left, z up) from camera frame with (x right, y down, z forward)
        # transform to robotics camera frame
        transform_vec = torch.tensor([1, -1, -1], device=device).unsqueeze(0).unsqueeze(2)
        pix_in_cam_frame = pix_in_cam_frame[:, [2, 0, 1], :] * transform_vec

        # normalize ray directions
        ray_directions = (pix_in_cam_frame / torch.norm(pix_in_cam_frame, dim=1, keepdim=True)).permute(0, 2, 1)
        # for camera, we always ray-cast from the sensor's origin
        ray_starts = torch.zeros_like(ray_directions, device=device)

        return ray_starts, ray_directions
    
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
            1,
        )
        self.depth_bev_input_pub = self.create_publisher(
            Image,
            self.depth_bev_input_topic,
            1,
        )
        self.forward_depth_embedding_pub = self.create_publisher(
            Float32MultiArray,
            self.forward_depth_embedding_topic,
            1,
        )
        self.get_logger().info("ros handlers started")



    def _init_bev_cache(self, H_img, W_img, cell_size_m, x_forward_m, y_halfwidth_m):
        # rays are constant per pixel; pre-cache
        rays = self.ray_directions[0].to(self.device)               # [HW,3]
        d_ax = rays[:, 0].contiguous()                               # axial = +X (your setup)
        good_ax = d_ax.abs() > 1e-8

        H_bev = int(math.ceil(x_forward_m / cell_size_m))
        W_bev = int(math.ceil((2.0 * y_halfwidth_m) / cell_size_m))
        H_bev = max(1, H_bev); W_bev = max(1, W_bev)

        self._bev_cache = {
            "H_img": H_img, "W_img": W_img,
            "rays": rays,                  # [HW,3]
            "d_ax": d_ax,                  # [HW]
            "good_ax": good_ax,            # [HW]
            "cell_size": cell_size_m,
            "x_max": x_forward_m, "y_half": y_halfwidth_m,
            "H_bev": H_bev, "W_bev": W_bev,
            "num_cells": H_bev * W_bev,
            "x_res": x_forward_m / H_bev,
            "y_res": (2.0 * y_halfwidth_m) / W_bev,
        }

    def depth_to_bev_batch(self,
                        depth_batch: torch.Tensor,  # [N,H_img,W_img], axial depth (can be negative; abs() used)
                        x_forward_m: float = 1.8,
                        y_halfwidth_m: float = 0.7,
                        height_clip_m: float = 1.0,
                        cell_size_m: float = 0.05) -> torch.Tensor:
        device = self.device
        N, H_img, W_img = depth_batch.shape
        HW = H_img * W_img

        # Init / refresh cache if missing or extents changed
        cache = getattr(self, "_bev_cache", None)
        if (cache is None
            or cache["H_img"] != H_img or cache["W_img"] != W_img
            or abs(cache["x_max"] - x_forward_m) > 1e-9
            or abs(cache["y_half"] - y_halfwidth_m) > 1e-9
            or abs(cache["cell_size"] - cell_size_m) > 1e-12):
            self._init_bev_cache(H_img, W_img, cell_size_m, x_forward_m, y_halfwidth_m)
            cache = self._bev_cache

        rays      = cache["rays"]         # [HW,3]
        d_ax      = cache["d_ax"]         # [HW]
        good_ax   = cache["good_ax"]      # [HW]
        H_bev     = cache["H_bev"]
        W_bev     = cache["W_bev"]
        num_cells = cache["num_cells"]
        x_res     = cache["x_res"]
        y_res     = cache["y_res"]

        # (1) axial depth → range along ray
        Zc = depth_batch.view(N, -1).abs()                     # [N,HW]
        valid = torch.isfinite(Zc) & (Zc > 0.0)                # [N,HW]

        # Safe divide (broadcast d_ax)
        denom = torch.where(d_ax.abs() > 1e-8, d_ax, torch.ones_like(d_ax))
        s = Zc / denom                                         # [N,HW]

        # (2) 3D camera points (broadcast rays)
        P_cam = s.unsqueeze(-1) * rays.unsqueeze(0)            # [N,HW,3]

        # (3) camera->body using per-env offsets (broadcast, no per-pixel expand)
        # t_cb: [N,3], q_cb: [N,4]  (camera pose in body frame)
        t_cb = self.camera_offsets                             # [N,3]
        q_cb = self.camera_quats_offsets                       # [N,4]
        # P_body = quat_rotate_inverse(q_cb[:, None, :], P_cam - t_cb[:, None, :])  # [N,HW,3]

        P_body = quat_rotate_inverse_batch(q_cb[:, None, :], P_cam - t_cb[:, None, :])  # [N,HW,3]


        X = P_body[..., 0]; Y = P_body[..., 1]; Z = P_body[..., 2]  # each [N,HW]

        # (4) bounds in body frame
        x_min, x_max = 0.0, x_forward_m
        y_min, y_max = -y_halfwidth_m, y_halfwidth_m

        in_bounds = (valid & good_ax.unsqueeze(0)
                    & (X >= x_min) & (X <= x_max)
                    & (Y >= y_min) & (Y <= y_max))                                # [N,HW]

        # Flatten indices of valid in-bounds points
        env_idx, pix_idx = torch.nonzero(in_bounds, as_tuple=True)                 # [K], [K]
        if env_idx.numel() == 0:
            # nothing visible for anyone
            bev = torch.zeros(N, H_bev, W_bev, device=device, dtype=torch.float32)
            return torch.clamp(bev, -height_clip_m, height_clip_m) / (2.0 * height_clip_m)

        # (5) bin to BEV
        Xs = X[env_idx, pix_idx]
        Ys = Y[env_idx, pix_idx]
        Zs = Z[env_idx, pix_idx]                         # heights (take per-cell MIN)

        ix = torch.clamp(((Xs - x_min) / x_res).floor().long(), 0, H_bev - 1)
        iy = torch.clamp(((Ys - y_min) / y_res).floor().long(), 0, W_bev - 1)
        flat_idx_local = ix * W_bev + iy                 # [K]

        # Offset per-env so we can scatter in one shot
        flat_idx_global = env_idx * num_cells + flat_idx_local  # [K]

        # (6) per-cell min via a single scatter_min over all envs
        K = Zs.numel()
        idx = flat_idx_global.to(torch.long).view(K)
        src = Zs.to(torch.float32).view(K)

        # Prefill output with +inf; unfilled cells will stay +inf
        bev_all_flat = torch.full((N * num_cells,), float('inf'),
                                device=src.device, dtype=src.dtype)

        # scatter_min will do: out[idx] = min(out[idx], src)
        bev_all_flat, _ = scatter_min(src, idx, dim=0, out=bev_all_flat)
        
        # (7) unknown -> 0, reshape, normalize
        hit_mask = torch.isfinite(bev_all_flat)
        bev_all_flat = torch.where(hit_mask, bev_all_flat, torch.zeros_like(bev_all_flat))

        bev = bev_all_flat.view(N, H_bev, W_bev)
        bev = torch.clamp(bev, -height_clip_m, height_clip_m) / (2.0 * height_clip_m)
        return bev  # [N,H_bev,W_bev]
    
    def get_bev_depth_frame(self):
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
        )                    # output shape is [106, 60]


        # --- 3. Convert to torch + batch dimension ---
        depth_batch = (
            torch.from_numpy(depth_image_np_resized)
            .float()
            .unsqueeze(0)     # N=1
            .to(self.device)
        )

        # Build BEV for everyone at once (abs() inside handles sign)
        bevs = self.depth_to_bev_batch(
            depth_batch = -depth_batch,       # make axial depth positive; abs() also applied inside
            x_forward_m = 1.8,   # 1.8
            y_halfwidth_m = 0.7,  # 0.7
            height_clip_m = 1.0,
            cell_size_m = 0.05
        )  # [N,H_bev,W_bev]


        # --- 5. Convert to CPU numpy ---
        bev_np = bevs[0].detach().cpu().numpy().astype(np.float32)
        # --- 6. Publish BEV image ---
        bev_msg = self.bridge.cv2_to_imgmsg(bev_np, encoding='32FC1')
        self.depth_bev_input_pub.publish(bev_msg)


        

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
        # depth_image_pyt = self.get_depth_frame()
        self.get_bev_depth_frame()



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

