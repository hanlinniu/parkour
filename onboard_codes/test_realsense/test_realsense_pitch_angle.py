import pyrealsense2 as rs
import math
import time

# Initialize pipeline and configuration
pipeline = rs.pipeline()
config = rs.config()

try:
    # Enable accelerometer and gyroscope streams
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)

    # Start the pipeline
    pipeline.start(config)
    print("Pipeline started successfully. Streaming IMU data...")

    # Variables to track yaw angle
    yaw = 0.0
    prev_time = time.time()

    while True:
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()

        # Get accelerometer data
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            accel_x, accel_y, accel_z = accel_data.x, accel_data.y, accel_data.z

            # Calculate roll and pitch angles
            roll = math.degrees(math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)))
            pitch = math.degrees(math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)))

        # Get gyroscope data
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            gyro_z = gyro_data.z

            # Calculate yaw angle by integrating gyroscope z-axis angular velocity
            current_time = time.time()
            dt = current_time - prev_time
            yaw += gyro_z * dt
            prev_time = current_time

        # Print roll, pitch, and yaw
        print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {math.degrees(yaw):.2f}°")

except Exception as e:
    print(f"Error: {e}")
finally:
    try:
        pipeline.stop()
        print("Pipeline stopped.")
    except RuntimeError:
        print("Pipeline was not started.")
