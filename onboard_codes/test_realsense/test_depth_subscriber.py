import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch
from cv_bridge import CvBridge


class DepthImageSubscriber(Node):
    def __init__(self):
        super().__init__('depth_image_subscriber')

        # ROS 2 Subscription
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/forward_depth',
            self.depth_callback,
            1  # QoS depth
        )

        # Buffer to store the latest depth message as a PyTorch tensor
        self.forward_depth_buffer = None

        # Device for PyTorch operations
        self.model_device = torch.device('cuda:0')

        # Bridge for ROS Image <-> OpenCV conversion
        self.bridge = CvBridge()

        self.get_logger().info("Subscribed to /camera/forward_depth")

    def depth_callback(self, msg):
        """Callback function for /camera/forward_depth topic."""
        self.forward_depth_buffer = torch.tensor(msg.data, device= self.model_device, dtype= torch.float32).view(1, -1)

    def _get_depth_obs(self):
        return self.forward_depth_buffer


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
