import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch
from cv_bridge import CvBridge
from test_depth_subscriber import DepthImageSubscriber
import time

class Go2Node(DepthImageSubscriber):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0')
        self.infos = {}
        self.infos["depth"] = None

        self.loop_counter = 0
        self.main_loop_timer = None  # Keep a reference to the timer
        self.last_update_time = None  # To calculate frequency
        

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
    def main_loop(self):
        # Increment the counter
        self.loop_counter += 1
        depth_data = self._get_depth_obs()

        # Update self.infos["depth"] every 5 steps
        if self.loop_counter % 5 == 0:
            # Update depth only if forward_depth_buffer is available
            if depth_data is not None:
                self.infos["depth"] = depth_data.clone()
                print("Depth data updated in infos")

                # Calculate frequency
                current_time = time.time()
                if self.last_update_time is not None:
                    time_diff = current_time - self.last_update_time
                    frequency = 1.0 / time_diff
                    print(f"Depth data frequency: {frequency:.2f} Hz")
                self.last_update_time = current_time
            else:
                self.infos["depth"] = None
                print("No depth data available to update infos")
        else:
            self.infos["depth"] = None
            print(f"Skipping update at loop count: {self.loop_counter}")

        # Print the state of depth data
        if self.infos["depth"] is None:
            print("0")
        else:
            print("1")
            print("There is depth data")

def main():
    rclpy.init()
    env_node = Go2Node()
    duration = 0.002
    env_node.start_main_loop_timer(duration)
    rclpy.spin(env_node)
    env_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()