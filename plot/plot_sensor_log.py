import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load Sensor Data ---
sensor_data = np.load("full_sensor_log.npy")  # shape: (N, 14)

# Unpack sensor data
step = sensor_data[:, 0]
gyro = sensor_data[:, 1:4]
imu_rpy = sensor_data[:, 4:7]
est_rpy = sensor_data[:, 7:10]
contact = sensor_data[:, 10:14]

# --- Plot Gyroscope ---
plt.figure()
plt.plot(step, gyro[:, 0], label="gyro_x")
plt.plot(step, gyro[:, 1], label="gyro_y")
plt.plot(step, gyro[:, 2], label="gyro_z")
plt.title("IMU Gyroscope")
plt.xlabel("Step Count")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid(True)

# --- Plot IMU RPY ---
plt.figure()
plt.plot(step, imu_rpy[:, 0], label="IMU roll")
plt.plot(step, imu_rpy[:, 1], label="IMU pitch")
plt.plot(step, imu_rpy[:, 2], label="IMU yaw")
plt.title("IMU RPY")
plt.xlabel("Step Count")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid(True)

# --- Plot Estimated RPY (from estimator or policy) ---
plt.figure()
plt.plot(step, est_rpy[:, 0], label="Est roll")
plt.plot(step, est_rpy[:, 1], label="Est pitch")
plt.plot(step, est_rpy[:, 2], label="Est yaw")
plt.title("Estimated RPY")
plt.xlabel("Step Count")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid(True)

# --- Plot Contact States ---
plt.figure()
for i in range(4):
    plt.plot(step, contact[:, i], label=f"contact_leg_{i}")
plt.title("Contact States")
plt.xlabel("Step Count")
plt.ylabel("Contact (0 or 1)")
plt.legend()
plt.grid(True)

# --- Optional: Yaw from obs log ---
if os.path.exists("yaw_log.npy"):
    yaw_data = np.load("yaw_log.npy")
    yaw_step = yaw_data[:, 0]
    yaw_0 = yaw_data[:, 1]
    yaw_1 = yaw_data[:, 2]

    plt.figure()
    plt.plot(yaw_step, yaw_0, label="yaw[0] (obs[:,6])")
    plt.plot(yaw_step, yaw_1, label="yaw[1] (obs[:,7])")
    plt.title("Yaw from obs[:,6:8]")
    plt.xlabel("Step Count")
    plt.ylabel("Yaw Values")
    plt.legend()
    plt.grid(True)
else:
    print("Note: 'yaw_log.npy' not found — skipping yaw plot.")

# Show all plots
try:
    plt.show()
except KeyboardInterrupt:
    print("\n[Interrupted] Closing plots...")
    plt.close('all')

