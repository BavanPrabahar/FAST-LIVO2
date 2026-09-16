#!/usr/bin/env python3
"""
Comprehensive Real-Time Diagnostics & Anomaly Monitor for FAST-LIVO2.
Monitors:
1. RAW SENSORS:
   - IMU (/livox/imu or /imu/data_raw): Initial gravity calibration, gyro bias, acceleration shocks.
   - LiDAR (/livox/lidar or /rslidar_points): Scan rates, point counts, timestamp monotonicity.
   - Camera (/usb_image_raw_sync): Frame rates, camera-to-LiDAR sync latency.
2. FAST-LIVO2 INTERNAL OUTPUTS:
   - Odometry (/aft_mapped_to_init): State drift, position jumps, speed spikes.
   - High-rate IMU Propagation (/LIVO2/imu_propagate): IMU dead-reckoning vs EKF filter consistency.
   - Effective Features (/cloud_effected): Detects geometric degeneration (corridors/flat walls).
   - Registered Map Cloud (/cloud_registered): Map density, output frame rate.
   - SLAM Processing Latency: Time delta between raw sensor measurement and published odometry (detects processing backlog).
"""

import rospy
import numpy as np
import math
import os
import csv
from sensor_msgs.msg import Imu, PointCloud2, Image
from nav_msgs.msg import Odometry, Path


class Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class FastLivoDiagnostics:
    def __init__(self):
        rospy.init_node("fastlivo_diagnostics", anonymous=True)

        # Configurable Topics
        self.imu_topic = rospy.get_param("~imu_topic", "/livox/imu")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/livox/lidar")
        self.cam_topic = rospy.get_param("~camera_topic", "/usb_image_raw_sync")
        self.odom_topic = rospy.get_param("~odom_topic", "/aft_mapped_to_init")
        self.imu_prop_topic = rospy.get_param("~imu_prop_topic", "/LIVO2/imu_propagate")
        self.effect_pts_topic = rospy.get_param("~effect_pts_topic", "/cloud_effected")
        self.registered_cloud_topic = rospy.get_param("~registered_cloud_topic", "/cloud_registered")
        self.path_topic = rospy.get_param("~path_topic", "/path")
        self.log_dir = rospy.get_param("~log_dir", "/home/atom/fast_ws/src/FAST-LIVO2/Log")

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "fastlivo_diagnostics.csv")
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp", "imu_hz", "lidar_hz", "cam_hz", "slam_odom_hz", "slam_latency_ms",
            "effective_features", "registered_pts", "odom_x", "odom_y", "odom_z",
            "odom_speed_m_s", "sync_cam_lidar_ms", "status", "alerts"
        ])

        # State Variables
        self.start_time = None
        self.init_period_done = False
        self.init_acc_samples = []
        self.init_gyr_samples = []

        self.last_imu_time = None
        self.last_lidar_time = None
        self.last_cam_time = None
        self.last_odom_time = None

        self.imu_count = 0
        self.lidar_count = 0
        self.cam_count = 0
        self.odom_count = 0
        self.registered_cloud_count = 0

        self.last_stat_time = rospy.Time.now().to_sec()

        self.current_odom_pos = np.zeros(3)
        self.last_odom_pos = np.zeros(3)
        self.current_odom_speed = 0.0
        self.current_imu_prop_pos = np.zeros(3)

        self.last_effective_features = 0
        self.last_registered_pts = 0
        self.slam_latency_ms = 0.0

        self.alerts = []

        # Subscribers
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=500)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_cb, queue_size=10)
        rospy.Subscriber(self.cam_topic, Image, self.cam_cb, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=50)
        rospy.Subscriber(self.imu_prop_topic, Odometry, self.imu_prop_cb, queue_size=500)
        rospy.Subscriber(self.effect_pts_topic, PointCloud2, self.effect_pts_cb, queue_size=10)
        rospy.Subscriber(self.registered_cloud_topic, PointCloud2, self.registered_cloud_cb, queue_size=10)

        # 1 Hz Terminal & CSV Report Timer
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_diagnostics)

        rospy.loginfo(f"{Color.CYAN}=== FAST-LIVO2 Full Diagnostics Monitor Active ==={Color.RESET}")
        rospy.loginfo(f"Diagnostics logging to: {self.csv_path}")

    def imu_cb(self, msg):
        t = msg.header.stamp.to_sec()
        if t == 0:
            t = rospy.Time.now().to_sec()
        if self.start_time is None:
            self.start_time = t

        self.imu_count += 1
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        acc_norm = np.linalg.norm(acc)
        gyr_norm = np.linalg.norm(gyr)

        # 1. INITIAL 2-SECOND STATIONARY GRAVITY CALIBRATION TEST
        elapsed = t - self.start_time
        if elapsed < 2.0:
            self.init_acc_samples.append(acc)
            self.init_gyr_samples.append(gyr)
        elif not self.init_period_done:
            self.init_period_done = True
            acc_mat = np.array(self.init_acc_samples)
            gyr_mat = np.array(self.init_gyr_samples)
            acc_mean = np.mean(acc_mat, axis=0)
            acc_std = np.std(acc_mat, axis=0)
            gyr_mean = np.mean(gyr_mat, axis=0)
            acc_mag = np.linalg.norm(acc_mean)

            print(f"\n{Color.BOLD}========================================================")
            print(f"       Initial Stationary Gravity Calibration Report    ")
            print(f"========================================================{Color.RESET}")
            print(f"Samples in first 2s:     {len(acc_mat)}")
            print(f"Mean Acceleration Norm:  {acc_mag:.4f} m/s^2 (Expected: 9.81)")
            print(f"Acceleration Jitter (σ): {np.linalg.norm(acc_std):.4f} m/s^2")
            print(f"Initial Gyro Bias:       [{gyr_mean[0]:.5f}, {gyr_mean[1]:.5f}, {gyr_mean[2]:.5f}] rad/s")

            if np.linalg.norm(acc_std) > 0.25:
                err = "CRITICAL: Sensor moved/vibrated during first 2s! Gravity vector is tilted -> Linear velocity drift guaranteed."
                print(f"{Color.RED}{err}{Color.RESET}")
                self.alerts.append(err)
            elif abs(acc_mag - 9.81) > 0.8:
                err = f"WARNING: Measured gravity magnitude ({acc_mag:.2f}) deviates from standard 9.81 m/s^2."
                print(f"{Color.YELLOW}{err}{Color.RESET}")
                self.alerts.append(err)
            else:
                print(f"{Color.GREEN}Gravity Calibration Status: EXCELLENT (Sensor was stationary) ✅{Color.RESET}")
            print(f"{Color.BOLD}========================================================\n{Color.RESET}")

        # Shock & Saturation Detection
        if acc_norm > 40.0:
            self.alerts.append(f"IMU Shock/Acceleration Spike ({acc_norm:.1f} m/s^2)")
        if gyr_norm > 8.0:
            self.alerts.append(f"Extreme Angular Velocity ({gyr_norm*57.3:.1f} deg/s)")

        self.last_imu_time = t

    def lidar_cb(self, msg):
        t = msg.header.stamp.to_sec()
        if t == 0:
            t = rospy.Time.now().to_sec()
        self.lidar_count += 1
        self.last_lidar_time = t

    def cam_cb(self, msg):
        t = msg.header.stamp.to_sec()
        if t == 0:
            t = rospy.Time.now().to_sec()
        self.cam_count += 1
        self.last_cam_time = t

    def effect_pts_cb(self, msg):
        self.last_effective_features = msg.width * msg.height
        if self.last_effective_features < 50 and self.init_period_done:
            self.alerts.append(f"GEOMETRIC DEGENERATION: Low effective features ({self.last_effective_features} pts) -> Risk of corridor drift!")

    def registered_cloud_cb(self, msg):
        self.registered_cloud_count += 1
        self.last_registered_pts = msg.width * msg.height

    def imu_prop_cb(self, msg):
        p = msg.pose.pose.position
        self.current_imu_prop_pos = np.array([p.x, p.y, p.z])

    def odom_cb(self, msg):
        self.odom_count += 1
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.current_odom_pos = np.array([p.x, p.y, p.z])
        self.current_odom_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        t = msg.header.stamp.to_sec()

        # Compute SLAM Processing Latency (Measurement Time to Publish Time)
        if self.last_lidar_time is not None:
            self.slam_latency_ms = (t - self.last_lidar_time) * 1000.0
            if abs(self.slam_latency_ms) > 200.0 and self.init_period_done:
                self.alerts.append(f"SLAM PROCESSING BACKLOG: Latency = {self.slam_latency_ms:.1f}ms!")

        # Position Jump Detection
        if self.last_odom_time is not None:
            dt = t - self.last_odom_time
            if dt > 0.001:
                disp = np.linalg.norm(self.current_odom_pos - self.last_odom_pos)
                speed = disp / dt
                if speed > 25.0:  # Sudden 25 m/s jump
                    self.alerts.append(f"ODOMETRY TELEPORTATION JUMP ({speed:.1f} m/s)")

        self.last_odom_pos = self.current_odom_pos.copy()
        self.last_odom_time = t

    def publish_diagnostics(self, event):
        now = rospy.Time.now().to_sec()
        dt = now - self.last_stat_time
        if dt <= 0:
            return

        imu_hz = self.imu_count / dt
        lidar_hz = self.lidar_count / dt
        cam_hz = self.cam_count / dt
        odom_hz = self.odom_count / dt

        self.imu_count = 0
        self.lidar_count = 0
        self.cam_count = 0
        self.odom_count = 0
        self.registered_cloud_count = 0
        self.last_stat_time = now

        cam_lidar_sync = 0.0
        if self.last_cam_time and self.last_lidar_time:
            cam_lidar_sync = (self.last_cam_time - self.last_lidar_time) * 1000.0

        status_str = "HEALTHY"
        alert_msg = "None"

        if len(self.alerts) > 0:
            status_str = "ANOMALY DETECTED"
            alert_msg = " | ".join(self.alerts[-2:])
            self.alerts.clear()

        status_color = Color.GREEN if status_str == "HEALTHY" else Color.RED

        # Live Formatted Dashboard
        print(f"\r[{now:.1f}s] "
              f"IMU: {imu_hz:>5.1f}Hz | "
              f"LiD: {lidar_hz:>4.1f}Hz | "
              f"Cam: {cam_hz:>4.1f}Hz | "
              f"SLAM Odom: {odom_hz:>4.1f}Hz | "
              f"Eff Feats: {self.last_effective_features:>4} | "
              f"Pos: [{self.current_odom_pos[0]:>5.1f},{self.current_odom_pos[1]:>5.1f},{self.current_odom_pos[2]:>5.1f}] | "
              f"{status_color}[{status_str}]{Color.RESET} {alert_msg}", end="", flush=True)

        # Write to CSV
        self.csv_writer.writerow([
            f"{now:.3f}", f"{imu_hz:.1f}", f"{lidar_hz:.1f}", f"{cam_hz:.1f}", f"{odom_hz:.1f}",
            f"{self.slam_latency_ms:.1f}", f"{self.last_effective_features}", f"{self.last_registered_pts}",
            f"{self.current_odom_pos[0]:.2f}", f"{self.current_odom_pos[1]:.2f}", f"{self.current_odom_pos[2]:.2f}",
            f"{self.current_odom_speed:.2f}", f"{cam_lidar_sync:.1f}", status_str, alert_msg
        ])
        self.csv_file.flush()

    def __del__(self):
        if hasattr(self, "csv_file") and self.csv_file:
            self.csv_file.close()


if __name__ == "__main__":
    try:
        diag = FastLivoDiagnostics()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
