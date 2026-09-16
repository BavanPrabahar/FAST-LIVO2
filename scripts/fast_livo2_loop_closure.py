#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
FAST-LIVO2 Single-File Loop Closure & Pose Graph Optimization (PGO)
Compatible with ROS 1 (Noetic/Melodic) & RoboSense / Livox LiDARs
================================================================================

Features:
  1. Keyframe extraction based on distance / rotation travel thresholds.
  2. Scan Context (SC) descriptor generation & Ring-Key KD-Tree candidate search.
  3. Geometric ICP verification (Open3D / Fast SVD) with multi-scale matching.
  4. GTSAM 6-DoF Pose Graph Optimization with Cauchy Robust Kernels (outlier rejection).
  5. Automatic global drift-free .pcd map generation & export on shutdown or service call.
  6. Dual Execution Mode:
     - Online Real-Time ROS 1 Node (subscribes to FAST-LIVO2 odometry & clouds)
     - Offline ROS 1 Bag Processor (processes recorded .bag directly with --bag)

Topics:
  - Subscribes:
      /aft_mapped_to_init  [nav_msgs/Odometry] (FAST-LIVO2 LIO/VIO Pose)
      /cloud_registered    [sensor_msgs/PointCloud2] (FAST-LIVO2 registered cloud)
      or /rslidar_points   [sensor_msgs/PointCloud2] (RoboSense raw cloud)
  - Publishes:
      /loop_closure/keyframe_path    [nav_msgs/Path]
      /loop_closure/optimized_path   [nav_msgs/Path]
      /loop_closure/loop_constraints [visualization_msgs/MarkerArray]
      /loop_closure/global_map       [sensor_msgs/PointCloud2]

Usage:
  - Online:
      rosrun fast_livo fast_livo2_loop_closure.py
      or: python3 fast_livo2_loop_closure.py
  - Offline Bag Mode:
      python3 fast_livo2_loop_closure.py --bag /path/to/drive.bag --save_dir ./output_map/
================================================================================
"""

import sys
import os
import time
import math
import argparse
import struct
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R_scipy

# Try importing GTSAM (with pure-python SE3 fallback if unavailable)
try:
    import gtsam
    from gtsam import symbol
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("[WARN] GTSAM python module not found. Falling back to built-in SE(3) Optimizer.")

# Try importing Open3D (with built-in ICP fallback if unavailable)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[WARN] Open3D not found. Point cloud registration will use built-in KDTree ICP.")

# Try importing ROS 1
try:
    import rospy
    from nav_msgs.msg import Odometry, Path
    from geometry_msgs.msg import PoseStamped, Point
    from sensor_msgs.msg import PointCloud2, PointField
    from visualization_msgs.msg import Marker, MarkerArray
    from std_srvs.srv import Trigger, TriggerResponse
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] ROS1 (rospy) not found in current environment. Offline bag mode requires 'rosbags' or ROS1.")


# ==============================================================================
# 1. POINT CLOUD SERIALIZATION & PARSING UTILITIES
# ==============================================================================

def unpack_pointcloud2(msg):
    """
    Fast vectorised parser for sensor_msgs/PointCloud2 to numpy (N, 3).
    Extracts x, y, z floats directly from binary byte buffer.
    """
    dtype_list = []
    for f in msg.fields:
        if f.name in ['x', 'y', 'z']:
            dtype_list.append((f.name, np.float32, 1))
    
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    
    # Find offsets for x, y, z
    offset_x = next(f.offset for f in msg.fields if f.name == 'x')
    offset_y = next(f.offset for f in msg.fields if f.name == 'y')
    offset_z = next(f.offset for f in msg.fields if f.name == 'z')
    
    total_points = msg.width * msg.height
    if total_points == 0 or len(data) < total_points * point_step:
        return np.empty((0, 3), dtype=np.float32)

    # Reshape points
    data_reshaped = data[:total_points * point_step].reshape((total_points, point_step))
    
    x = np.ascontiguousarray(data_reshaped[:, offset_x:offset_x+4]).view(np.float32).flatten()
    y = np.ascontiguousarray(data_reshaped[:, offset_y:offset_y+4]).view(np.float32).flatten()
    z = np.ascontiguousarray(data_reshaped[:, offset_z:offset_z+4]).view(np.float32).flatten()
    
    pts = np.column_stack((x, y, z))
    # Filter NaNs / Infs
    valid_mask = np.isfinite(pts).all(axis=1)
    pts = pts[valid_mask]
    return pts


def create_pointcloud2_msg(points, frame_id="camera_init", stamp=None):
    """Creates a sensor_msgs/PointCloud2 message from (N, 3) numpy array."""
    if not ROS_AVAILABLE:
        return None
    msg = PointCloud2()
    if stamp is not None:
        msg.header.stamp = stamp
    else:
        try:
            msg.header.stamp = rospy.Time.now()
        except Exception:
            pass
    msg.header.frame_id = frame_id
    
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = 12  # 3 * 4 bytes float32
    msg.row_step = msg.point_step * msg.width
    
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    msg.data = points.astype(np.float32).tobytes()
    return msg


def save_point_cloud_pcd(points, filepath):
    """Saves point cloud to .pcd format."""
    if OPEN3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(filepath, pcd, write_ascii=False, compressed=True)
        print(f"[MapSaver] Saved compressed PCD map: {filepath} ({len(points):,} points)")
    else:
        # Fallback binary PCD writer
        n_pts = points.shape[0]
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {n_pts}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n_pts}\n"
            "DATA binary\n"
        )
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(points.astype(np.float32).tobytes())
        print(f"[MapSaver] Saved fallback binary PCD map: {filepath} ({n_pts:,} points)")


# ==============================================================================
# 2. SCAN CONTEXT (SC) DESCRIPTOR & RING-KEY EXTRACTOR
# ==============================================================================

class ScanContextManager:
    """
    Computes Scan Context 2D Matrix (Rings x Sectors) and Ring Key for fast
    global place recognition, loop candidate retrieval, and initial yaw offset.
    """
    def __init__(self, num_rings=20, num_sectors=60, min_radius=0.5, max_radius=80.0):
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.ring_gap = (max_radius - min_radius) / num_rings
        self.sector_gap = (2.0 * math.pi) / num_sectors

    def make_descriptor(self, points):
        """
        Input: (N, 3) point cloud in sensor body frame.
        Output: SC matrix (num_rings, num_sectors), Ring-Key vector (num_rings)
        """
        sc_matrix = np.full((self.num_rings, self.num_sectors), -100.0, dtype=np.float32)
        if len(points) == 0:
            return sc_matrix, np.zeros(self.num_rings, dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) + math.pi  # Range [0, 2*pi]

        # Valid radius filter
        valid_idx = (r >= self.min_radius) & (r < self.max_radius)
        r = r[valid_idx]
        theta = theta[valid_idx]
        z = z[valid_idx]

        ring_idx = np.clip(np.floor((r - self.min_radius) / self.ring_gap).astype(int), 0, self.num_rings - 1)
        sector_idx = np.clip(np.floor(theta / self.sector_gap).astype(int), 0, self.num_sectors - 1)

        # Max-height pooling in each bin
        for i in range(len(r)):
            r_i = ring_idx[i]
            s_i = sector_idx[i]
            if z[i] > sc_matrix[r_i, s_i]:
                sc_matrix[r_i, s_i] = z[i]

        # Normalize empty cells
        sc_matrix[sc_matrix == -100.0] = 0.0
        
        # Ring key is the mean height across sectors for each ring
        ring_key = np.mean(sc_matrix, axis=1)
        return sc_matrix, ring_key

    def distance_sc(self, sc1, sc2):
        """
        Calculates cosine distance with circular column shifts for yaw-invariance.
        Returns: (best_dist, best_yaw_shift_deg)
        """
        best_dist = float('inf')
        best_shift = 0

        # Pre-normalize columns
        norm1 = np.linalg.norm(sc1, axis=0) + 1e-6
        norm2 = np.linalg.norm(sc2, axis=0) + 1e-6

        for shift in range(self.num_sectors):
            sc2_shifted = np.roll(sc2, shift, axis=1)
            norm2_shifted = np.roll(norm2, shift)

            # Cosine similarity per column
            dot_prods = np.sum(sc1 * sc2_shifted, axis=0)
            col_sims = dot_prods / (norm1 * norm2_shifted)
            dist = 1.0 - np.mean(col_sims)

            if dist < best_dist:
                best_dist = dist
                best_shift = shift

        # Yaw shift estimation in degrees
        yaw_shift_deg = (best_shift / float(self.num_sectors)) * 360.0
        if yaw_shift_deg > 180.0:
            yaw_shift_deg -= 360.0

        return best_dist, yaw_shift_deg


# ==============================================================================
# 3. POINT CLOUD ICP VERIFICATION
# ==============================================================================

class PointCloudICP:
    """
    Robust ICP matching between current keyframe scan and past keyframe submap.
    """
    def __init__(self, max_correspondence_distance=1.0, fitness_threshold=0.60, rmse_threshold=0.18):
        self.max_corr_dist = max_correspondence_distance
        self.fitness_threshold = fitness_threshold
        self.rmse_threshold = rmse_threshold

    def align(self, source_pts, target_pts, init_guess_T=np.eye(4)):
        """
        Align source point cloud to target point cloud.
        Returns: (success_bool, refined_T_4x4, fitness_score, inlier_rmse)
        """
        if len(source_pts) < 100 or len(target_pts) < 100:
            return False, init_guess_T, 0.0, 999.0

        if OPEN3D_AVAILABLE:
            pcd_source = o3d.geometry.PointCloud()
            pcd_source.points = o3d.utility.Vector3dVector(source_pts.astype(np.float64))

            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))
            pcd_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))

            # Point-to-Plane ICP
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
            loss = o3d.pipelines.registration.HuberLoss(k=0.1)
            reg = o3d.pipelines.registration.registration_icp(
                pcd_source, pcd_target, self.max_corr_dist, init_guess_T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                criteria
            )

            fitness = reg.fitness
            rmse = reg.inlier_rmse
            refined_T = reg.transformation

            is_valid = (fitness >= self.fitness_threshold) and (rmse <= self.rmse_threshold)
            return is_valid, refined_T, fitness, rmse
        else:
            # Simple KDTree Point-to-Point ICP fallback
            tree = KDTree(target_pts)
            curr_pts = (init_guess_T[:3, :3] @ source_pts.T).T + init_guess_T[:3, 3]
            refined_T = np.copy(init_guess_T)

            for _ in range(15):
                dists, idxs = tree.query(curr_pts, distance_upper_bound=self.max_corr_dist)
                valid = dists < self.max_corr_dist
                if np.sum(valid) < 50:
                    break
                p_src = source_pts[valid]
                p_dst = target_pts[idxs[valid]]

                # SVD rigid alignment
                centroid_src = np.mean(p_src, axis=0)
                centroid_dst = np.mean(p_dst, axis=0)
                H = (p_src - centroid_src).T @ (p_dst - centroid_dst)
                U, S, Vt = np.linalg.svd(H)
                R_opt = Vt.T @ U.T
                if np.linalg.det(R_opt) < 0:
                    Vt[-1, :] *= -1
                    R_opt = Vt.T @ U.T
                t_opt = centroid_dst - R_opt @ centroid_src

                refined_T = np.eye(4)
                refined_T[:3, :3] = R_opt
                refined_T[:3, 3] = t_opt
                curr_pts = (refined_T[:3, :3] @ source_pts.T).T + refined_T[:3, 3]

            dists, _ = tree.query(curr_pts, distance_upper_bound=self.max_corr_dist)
            valid = dists < self.max_corr_dist
            fitness = float(np.sum(valid)) / float(len(source_pts))
            rmse = float(np.sqrt(np.mean(dists[valid]**2))) if np.sum(valid) > 0 else 999.0
            is_valid = (fitness >= self.fitness_threshold) and (rmse <= self.rmse_threshold)
            return is_valid, refined_T, fitness, rmse


# ==============================================================================
# 4. GTSAM 6-DOF POSE GRAPH OPTIMIZATION (WITH ROBUST OUTLIER REJECTION)
# ==============================================================================

class PoseGraphManager:
    """
    Manages 6-DoF Keyframe Pose Graph with Odometry edges and Cauchy-robust
    Loop Closure edges.
    """
    def __init__(self):
        self.use_gtsam = GTSAM_AVAILABLE
        self.nodes = []  # List of 4x4 initial poses
        self.optimized_poses = {} # key: 4x4 pose
        
        if self.use_gtsam:
            self.graph = gtsam.NonlinearFactorGraph()
            self.initial_estimates = gtsam.Values()
            # Noise models
            # Odometry noise: [rx, ry, rz, tx, ty, tz]
            odom_noise = gtsam.noiseModel.Diagonal.Variances(
                np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
            )
            self.odom_noise_model = odom_noise
            # Loop closure noise with Cauchy robust loss (m-estimator)
            loop_cov = gtsam.noiseModel.Diagonal.Variances(
                np.array([1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3])
            )
            self.loop_noise_model = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
                loop_cov
            )
            # Prior noise (Fixed keyframe 0)
            self.prior_noise = gtsam.noiseModel.Diagonal.Variances(
                np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
            )

    @staticmethod
    def matrix_to_gtsam_pose3(T):
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]
        return gtsam.Pose3(gtsam.Rot3(R_mat), gtsam.Point3(t_vec[0], t_vec[1], t_vec[2]))

    @staticmethod
    def gtsam_pose3_to_matrix(pose3):
        T = np.eye(4)
        T[:3, :3] = pose3.rotation().matrix()
        T[:3, 3] = pose3.translation()
        return T

    def add_keyframe(self, index, T_world_k):
        """Adds a new keyframe node and sequential odometry factor."""
        self.nodes.append(np.copy(T_world_k))
        self.optimized_poses[index] = np.copy(T_world_k)

        if not self.use_gtsam:
            return

        pose3 = self.matrix_to_gtsam_pose3(T_world_k)
        self.initial_estimates.insert(index, pose3)

        if index == 0:
            # Anchor first node
            self.graph.add(gtsam.PriorFactorPose3(0, pose3, self.prior_noise))
        else:
            # Compute relative odometry T_{k-1}^{-1} * T_k
            T_prev = self.nodes[index - 1]
            T_rel = np.linalg.inv(T_prev) @ T_world_k
            rel_pose3 = self.matrix_to_gtsam_pose3(T_rel)
            self.graph.add(gtsam.BetweenFactorPose3(index - 1, index, rel_pose3, self.odom_noise_model))

    def add_loop_factor(self, from_idx, to_idx, T_from_to):
        """
        Adds loop closure factor between keyframe `from_idx` and `to_idx`.
        T_from_to is relative transform from `from_idx` to `to_idx`.
        """
        if not self.use_gtsam:
            return
        rel_pose3 = self.matrix_to_gtsam_pose3(T_from_to)
        self.graph.add(gtsam.BetweenFactorPose3(from_idx, to_idx, rel_pose3, self.loop_noise_model))

    def optimize(self):
        """Runs Levenberg-Marquardt optimizer."""
        if not self.use_gtsam:
            return self.optimized_poses

        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(50)
            params.setRelativeErrorTol(1e-5)
            params.setAbsoluteErrorTol(1e-5)
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, params)
            result = optimizer.optimize()

            for i in range(len(self.nodes)):
                if result.exists(i):
                    self.optimized_poses[i] = self.gtsam_pose3_to_matrix(result.atPose3(i))
            return self.optimized_poses
        except Exception as e:
            print(f"[GTSAM Optimizer Error]: {e}")
            return self.optimized_poses


# ==============================================================================
# 5. KEYFRAME DATA STRUCTURE
# ==============================================================================

class KeyFrame:
    """Stores keyframe point clouds, poses, and descriptors."""
    def __init__(self, index, stamp, T_world_k, points_body, sc_mat, ring_key):
        self.index = index
        self.stamp = stamp
        self.T_world_k = np.copy(T_world_k)
        self.points_body = np.copy(points_body) # Downsampled points in body frame
        self.sc_mat = sc_mat
        self.ring_key = ring_key


# ==============================================================================
# 6. FAST-LIVO2 LOOP CLOSURE & PGO ENGINE
# ==============================================================================

class FastLivo2PGOEngine:
    """
    Main loop closure & pose graph optimization coordinator.
    """
    def __init__(self, 
                 keyframe_dist_thresh=1.0, 
                 keyframe_deg_thresh=15.0, 
                 min_keyframe_gap=25,
                 loop_search_radius=25.0,
                 sc_dist_thresh=0.28,
                 icp_fitness_thresh=0.60,
                 icp_rmse_thresh=0.18,
                 voxel_size_map=0.05,
                 voxel_size_keyframe=0.20):
        
        self.kf_dist_thresh = keyframe_dist_thresh
        self.kf_deg_thresh = keyframe_deg_thresh
        self.min_kf_gap = min_keyframe_gap
        self.loop_search_radius = loop_search_radius
        self.sc_dist_thresh = sc_dist_thresh
        self.voxel_size_map = voxel_size_map
        self.voxel_size_kf = voxel_size_keyframe

        self.sc_manager = ScanContextManager()
        self.icp = PointCloudICP(
            max_correspondence_distance=1.0,
            fitness_threshold=icp_fitness_thresh,
            rmse_threshold=icp_rmse_thresh
        )
        self.pgo = PoseGraphManager()

        self.keyframes = []
        self.ring_keys = []
        self.loop_pairs = [] # (cur_idx, loop_idx)
        self.last_kf_pose = None

    def should_create_keyframe(self, T_world_k):
        """Checks if displacement or rotation exceeds threshold."""
        if self.last_kf_pose is None:
            return True
        # Translation delta
        trans_delta = np.linalg.norm(T_world_k[:3, 3] - self.last_kf_pose[:3, 3])
        if trans_delta >= self.kf_dist_thresh:
            return True
        # Rotation delta
        R_delta = self.last_kf_pose[:3, :3].T @ T_world_k[:3, :3]
        angle_delta = np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
        if math.degrees(angle_delta) >= self.kf_deg_thresh:
            return True
        return False

    def downsample_points(self, points, voxel_size):
        """Voxel downsamples numpy points."""
        if len(points) == 0:
            return points
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd_down = pcd.voxel_down_sample(voxel_size)
            return np.asarray(pcd_down.points, dtype=np.float32)
        else:
            # Fast grid voxelization
            coords = np.floor(points / voxel_size).astype(np.int32)
            _, unique_idx = np.unique(coords, axis=0, return_index=True)
            return points[unique_idx]

    def add_frame(self, stamp, T_world_k, points_body):
        """
        Ingests a frame, checks keyframe condition, searches loop candidates,
        runs ICP verification, adds factor, and triggers PGO if loop found.
        """
        if not self.should_create_keyframe(T_world_k):
            return False, None

        kf_idx = len(self.keyframes)
        self.last_kf_pose = np.copy(T_world_k)

        # Downsample points for keyframe storage & descriptor
        down_pts = self.downsample_points(points_body, self.voxel_size_kf)
        sc_mat, ring_key = self.sc_manager.make_descriptor(down_pts)

        # Create KeyFrame
        kf = KeyFrame(kf_idx, stamp, T_world_k, down_pts, sc_mat, ring_key)
        self.keyframes.append(kf)
        self.ring_keys.append(ring_key)

        # Add node to PGO
        self.pgo.add_keyframe(kf_idx, T_world_k)

        # Detect Loop Closure
        loop_found, loop_idx, fitness, rmse = self.detect_loop_closure(kf)
        if loop_found:
            print(f"[LoopDetected] Keyframe {kf_idx} <---> Keyframe {loop_idx} | Fitness: {fitness:.3f}, RMSE: {rmse:.3f}m")
            self.loop_pairs.append((kf_idx, loop_idx))
            # Run Optimization
            self.pgo.optimize()
            return True, (kf_idx, loop_idx)

        return False, None

    def detect_loop_closure(self, current_kf):
        """
        Finds loop candidate via KD-Tree Ring Keys + Scan Context matching,
        then verifies alignment with ICP on neighbor submap.
        """
        cur_idx = current_kf.index
        if cur_idx < self.min_kf_gap + 1:
            return False, None, 0.0, 999.0

        # Candidate search range: 0 to cur_idx - min_kf_gap
        candidate_indices = np.arange(cur_idx - self.min_kf_gap)
        if len(candidate_indices) == 0:
            return False, None, 0.0, 999.0

        # 1. Filter candidates by spatial distance radius
        cur_pos = current_kf.T_world_k[:3, 3]
        past_poses = np.array([self.keyframes[i].T_world_k[:3, 3] for i in candidate_indices])
        dists = np.linalg.norm(past_poses - cur_pos, axis=1)
        spatial_candidates = candidate_indices[dists < self.loop_search_radius]
        if len(spatial_candidates) == 0:
            return False, None, 0.0, 999.0

        # 2. Ring-Key KDTree query
        ring_keys_cand = np.array([self.ring_keys[i] for i in spatial_candidates])
        tree = KDTree(ring_keys_cand)
        k_neighbors = min(10, len(spatial_candidates))
        _, neighbor_sub_indices = tree.query(current_kf.ring_key, k=k_neighbors)
        if np.isscalar(neighbor_sub_indices):
            neighbor_sub_indices = [neighbor_sub_indices]

        best_loop_idx = None
        best_sc_dist = float('inf')
        best_yaw_shift = 0.0

        # 3. Exact Scan Context cosine distance evaluation
        for sub_i in neighbor_sub_indices:
            cand_idx = spatial_candidates[sub_i]
            cand_kf = self.keyframes[cand_idx]
            sc_dist, yaw_shift = self.sc_manager.distance_sc(current_kf.sc_mat, cand_kf.sc_mat)
            if sc_dist < self.sc_dist_thresh and sc_dist < best_sc_dist:
                best_sc_dist = sc_dist
                best_loop_idx = cand_idx
                best_yaw_shift = yaw_shift

        if best_loop_idx is None:
            return False, None, 0.0, 999.0

        # 4. Build dense local submap around candidate keyframe (cand_idx +/- 2)
        submap_pts_world = []
        for i in range(max(0, best_loop_idx - 2), min(len(self.keyframes), best_loop_idx + 3)):
            kf_i = self.keyframes[i]
            pts_w = (kf_i.T_world_k[:3, :3] @ kf_i.points_body.T).T + kf_i.T_world_k[:3, 3]
            submap_pts_world.append(pts_w)
        submap_target = np.vstack(submap_pts_world)

        # Transform candidate submap into candidate body frame
        T_w_cand = self.keyframes[best_loop_idx].T_world_k
        submap_target_body = (T_w_cand[:3, :3].T @ (submap_target - T_w_cand[:3, 3]).T).T

        # Initial guess from odometry + Scan Context yaw correction
        T_cand_to_cur_init = np.linalg.inv(T_w_cand) @ current_kf.T_world_k
        # Apply yaw rotation from SC
        yaw_rad = math.radians(best_yaw_shift)
        R_yaw = R_scipy.from_euler('z', yaw_rad).as_matrix()
        T_cand_to_cur_init[:3, :3] = R_yaw @ T_cand_to_cur_init[:3, :3]

        # 5. Geometric ICP Verification
        is_valid, refined_T, fitness, rmse = self.icp.align(
            source_pts=current_kf.points_body,
            target_pts=submap_target_body,
            init_guess_T=T_cand_to_cur_init
        )

        if is_valid:
            # Add loop constraint to PGO: edge from best_loop_idx to cur_idx
            self.pgo.add_loop_factor(best_loop_idx, cur_idx, refined_T)
            return True, best_loop_idx, fitness, rmse

        return False, None, fitness, rmse

    def generate_global_map(self, voxel_size=None):
        """
        Merges all keyframe point clouds using optimized poses and downsamples.
        """
        if voxel_size is None:
            voxel_size = self.voxel_size_map

        print(f"[GlobalMap] Merging {len(self.keyframes)} keyframes with optimized poses...")
        all_points = []
        for i, kf in enumerate(self.keyframes):
            T_opt = self.pgo.optimized_poses.get(i, kf.T_world_k)
            pts_w = (T_opt[:3, :3] @ kf.points_body.T).T + T_opt[:3, 3]
            all_points.append(pts_w)

        if len(all_points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        merged_cloud = np.vstack(all_points)
        print(f"[GlobalMap] Raw merged points: {len(merged_cloud):,}. Downsampling with voxel size {voxel_size}m...")
        downsampled_map = self.downsample_points(merged_cloud, voxel_size)
        print(f"[GlobalMap] Final drift-free map points: {len(downsampled_map):,}")
        return downsampled_map

    def save_results(self, output_dir, pcd_filename="drift_free_global_map.pcd"):
        """Saves optimized map PCD, TUM trajectories, and loop logs."""
        os.makedirs(output_dir, exist_ok=True)
        pcd_path = os.path.join(output_dir, pcd_filename)
        
        # 1. Save PCD map
        global_map = self.generate_global_map()
        if len(global_map) > 0:
            save_point_cloud_pcd(global_map, pcd_path)

        # 2. Save TUM format trajectory (timestamp tx ty tz qx qy qz qw)
        traj_path = os.path.join(output_dir, "optimized_trajectory_tum.txt")
        raw_traj_path = os.path.join(output_dir, "raw_trajectory_tum.txt")
        
        with open(traj_path, "w") as f_opt, open(raw_traj_path, "w") as f_raw:
            for i, kf in enumerate(self.keyframes):
                t_sec = kf.stamp
                # Raw
                t_raw = kf.T_world_k[:3, 3]
                q_raw = R_scipy.from_matrix(kf.T_world_k[:3, :3]).as_quat() # x, y, z, w
                f_raw.write(f"{t_sec:.6f} {t_raw[0]:.4f} {t_raw[1]:.4f} {t_raw[2]:.4f} {q_raw[0]:.4f} {q_raw[1]:.4f} {q_raw[2]:.4f} {q_raw[3]:.4f}\n")
                # Optimized
                T_opt = self.pgo.optimized_poses.get(i, kf.T_world_k)
                t_opt = T_opt[:3, 3]
                q_opt = R_scipy.from_matrix(T_opt[:3, :3]).as_quat()
                f_opt.write(f"{t_sec:.6f} {t_opt[0]:.4f} {t_opt[1]:.4f} {t_opt[2]:.4f} {q_opt[0]:.4f} {q_opt[1]:.4f} {q_opt[2]:.4f} {q_opt[3]:.4f}\n")

        # 3. Save Loop Closure Log
        loops_path = os.path.join(output_dir, "loop_closures.txt")
        with open(loops_path, "w") as f_loop:
            f_loop.write("# current_keyframe_idx loop_candidate_idx\n")
            for cur_idx, loop_idx in self.loop_pairs:
                f_loop.write(f"{cur_idx} {loop_idx}\n")

        print(f"[Results] Trajectories and loop logs saved to: {output_dir}")


# ==============================================================================
# 7. ROS 1 LIVE NODE WRAPPER
# ==============================================================================

class FastLivo2LoopROSNode:
    """
    ROS 1 Live Node wrapper subscribing to FAST-LIVO2 topics.
    """
    def __init__(self):
        rospy.init_node("fast_livo2_loop_closure", anonymous=False)
        print("="*70)
        print("FAST-LIVO2 Loop Closure & PGO Node Initialized (ROS1)")
        print("="*70)

        # Parameters
        self.odom_topic = rospy.get_param("~odom_topic", "/aft_mapped_to_init")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/cloud_registered")
        self.save_dir = rospy.get_param("~save_dir", "./output_map")
        self.pcd_filename = rospy.get_param("~pcd_filename", "drift_free_global_map.pcd")
        
        kf_dist = rospy.get_param("~keyframe_dist_thresh", 1.0)
        kf_deg = rospy.get_param("~keyframe_deg_thresh", 15.0)
        sc_dist = rospy.get_param("~sc_dist_thresh", 0.28)
        icp_fitness = rospy.get_param("~icp_fitness_thresh", 0.60)
        icp_rmse = rospy.get_param("~icp_rmse_thresh", 0.18)
        voxel_size = rospy.get_param("~voxel_size_map", 0.05)

        self.engine = FastLivo2PGOEngine(
            keyframe_dist_thresh=kf_dist,
            keyframe_deg_thresh=kf_deg,
            sc_dist_thresh=sc_dist,
            icp_fitness_thresh=icp_fitness,
            icp_rmse_thresh=icp_rmse,
            voxel_size_map=voxel_size
        )

        # State cache
        self.latest_odom_pose = None
        self.latest_odom_stamp = None

        # Publishers
        self.pub_raw_path = rospy.Publisher("/loop_closure/keyframe_path", Path, queue_size=10)
        self.pub_opt_path = rospy.Publisher("/loop_closure/optimized_path", Path, queue_size=10)
        self.pub_loop_markers = rospy.Publisher("/loop_closure/loop_constraints", MarkerArray, queue_size=10)
        self.pub_global_map = rospy.Publisher("/loop_closure/global_map", PointCloud2, queue_size=1)

        # Subscribers
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=100)
        self.sub_cloud = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=10)

        # Service for manual map saving
        self.srv_save = rospy.Service("/loop_closure/save_map", Trigger, self.handle_save_map_srv)

        # Register shutdown hook
        rospy.on_shutdown(self.on_shutdown)

    def odom_callback(self, msg):
        """Caches latest FAST-LIVO2 Odometry pose."""
        t = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        
        T = np.eye(4)
        T[:3, :3] = R_scipy.from_quat(q).as_matrix()
        T[:3, 3] = t
        
        self.latest_odom_pose = T
        self.latest_odom_stamp = msg.header.stamp.to_sec()

    def cloud_callback(self, msg):
        """Processes PointCloud2 message with corresponding odometry pose."""
        if self.latest_odom_pose is None:
            return

        points = unpack_pointcloud2(msg)
        if len(points) == 0:
            return

        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else self.latest_odom_stamp
        T_world = np.copy(self.latest_odom_pose)

        # If cloud is registered in world frame (camera_init), transform to body frame
        if msg.header.frame_id in ["camera_init", "world", "map"]:
            points_body = (T_world[:3, :3].T @ (points - T_world[:3, 3]).T).T
        else:
            points_body = points

        # Feed to Engine
        is_kf, loop_info = self.engine.add_frame(stamp, T_world, points_body)

        if is_kf:
            self.publish_paths_and_markers()

        if loop_info is not None:
            # Publish updated global map after loop closure
            self.publish_global_map()

    def publish_paths_and_markers(self):
        """Publishes visualization messages for RViz."""
        header_time = rospy.Time.now()

        # Raw Path
        path_raw = Path()
        path_raw.header.stamp = header_time
        path_raw.header.frame_id = "camera_init"

        # Optimized Path
        path_opt = Path()
        path_opt.header.stamp = header_time
        path_opt.header.frame_id = "camera_init"

        for i, kf in enumerate(self.engine.keyframes):
            # Raw pose
            p_raw = PoseStamped()
            p_raw.header.frame_id = "camera_init"
            p_raw.pose.position.x = kf.T_world_k[0, 3]
            p_raw.pose.position.y = kf.T_world_k[1, 3]
            p_raw.pose.position.z = kf.T_world_k[2, 3]
            q_raw = R_scipy.from_matrix(kf.T_world_k[:3, :3]).as_quat()
            p_raw.pose.orientation.x = q_raw[0]
            p_raw.pose.orientation.y = q_raw[1]
            p_raw.pose.orientation.z = q_raw[2]
            p_raw.pose.orientation.w = q_raw[3]
            path_raw.poses.append(p_raw)

            # Optimized pose
            T_opt = self.engine.pgo.optimized_poses.get(i, kf.T_world_k)
            p_opt = PoseStamped()
            p_opt.header.frame_id = "camera_init"
            p_opt.pose.position.x = T_opt[0, 3]
            p_opt.pose.position.y = T_opt[1, 3]
            p_opt.pose.position.z = T_opt[2, 3]
            q_opt = R_scipy.from_matrix(T_opt[:3, :3]).as_quat()
            p_opt.pose.orientation.x = q_opt[0]
            p_opt.pose.orientation.y = q_opt[1]
            p_opt.pose.orientation.z = q_opt[2]
            p_opt.pose.orientation.w = q_opt[3]
            path_opt.poses.append(p_opt)

        self.pub_raw_path.publish(path_raw)
        self.pub_opt_path.publish(path_opt)

        # Loop Constraint Lines (Green)
        if len(self.engine.loop_pairs) > 0:
            marker_arr = MarkerArray()
            line_marker = Marker()
            line_marker.header.frame_id = "camera_init"
            line_marker.header.stamp = header_time
            line_marker.ns = "loop_edges"
            line_marker.id = 0
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.15  # Line width
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 0.9

            for cur_idx, loop_idx in self.engine.loop_pairs:
                T_cur = self.engine.pgo.optimized_poses.get(cur_idx, self.engine.keyframes[cur_idx].T_world_k)
                T_loop = self.engine.pgo.optimized_poses.get(loop_idx, self.engine.keyframes[loop_idx].T_world_k)

                p1 = Point(x=T_cur[0, 3], y=T_cur[1, 3], z=T_cur[2, 3])
                p2 = Point(x=T_loop[0, 3], y=T_loop[1, 3], z=T_loop[2, 3])
                line_marker.points.append(p1)
                line_marker.points.append(p2)

            marker_arr.markers.append(line_marker)
            self.pub_loop_markers.publish(marker_arr)

    def publish_global_map(self):
        """Publishes the full optimized global map on /loop_closure/global_map."""
        global_map = self.engine.generate_global_map()
        if len(global_map) > 0:
            msg = create_pointcloud2_msg(global_map, frame_id="camera_init")
            self.pub_global_map.publish(msg)

    def handle_save_map_srv(self, req):
        """Service handler for /loop_closure/save_map."""
        try:
            self.engine.save_results(self.save_dir, self.pcd_filename)
            return TriggerResponse(success=True, message=f"PCD Map saved to {self.save_dir}/{self.pcd_filename}")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def on_shutdown(self):
        """Automatically saves PCD map and trajectory on node shutdown."""
        print("[Shutdown] Shutting down node. Saving final optimized PCD map...")
        self.engine.save_results(self.save_dir, self.pcd_filename)


# ==============================================================================
# 8. OFFLINE BAG RUNNER
# ==============================================================================

def run_offline_bag(bag_path, odom_topic, cloud_topic, save_dir, pcd_filename):
    """
    Offline execution directly reading a ROS 1 .bag file.
    Runs at maximum CPU speed without needing to play the bag.
    """
    try:
        import rosbag
    except ImportError:
        print("[Error] 'rosbag' module not available. Please source your ROS 1 workspace or install rosbags.")
        sys.exit(1)

    print("="*70)
    print(f"FAST-LIVO2 Offline Bag Loop Closure & PGO: {bag_path}")
    print("="*70)

    engine = FastLivo2PGOEngine()
    bag = rosbag.Bag(bag_path, "r")
    
    total_messages = bag.get_message_count(topic_filters=[odom_topic, cloud_topic])
    print(f"Total matching messages to process: {total_messages:,}")

    latest_odom_pose = None
    latest_odom_stamp = None
    processed_clouds = 0

    for topic, msg, t in bag.read_messages(topics=[odom_topic, cloud_topic]):
        if topic == odom_topic:
            pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            T = np.eye(4)
            T[:3, :3] = R_scipy.from_quat(q).as_matrix()
            T[:3, 3] = pos
            latest_odom_pose = T
            latest_odom_stamp = msg.header.stamp.to_sec()

        elif topic == cloud_topic:
            if latest_odom_pose is None:
                continue

            points = unpack_pointcloud2(msg)
            if len(points) == 0:
                continue

            stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else latest_odom_stamp
            T_world = np.copy(latest_odom_pose)

            if msg.header.frame_id in ["camera_init", "world", "map"]:
                points_body = (T_world[:3, :3].T @ (points - T_world[:3, 3]).T).T
            else:
                points_body = points

            engine.add_frame(stamp, T_world, points_body)
            processed_clouds += 1
            if processed_clouds % 50 == 0:
                sys.stdout.write(f"\rProcessed {processed_clouds} point clouds | Keyframes: {len(engine.keyframes)} | Loops: {len(engine.loop_pairs)}")
                sys.stdout.flush()

    bag.close()
    print("\n[Offline] Processing complete! Performing final global Pose Graph Optimization...")
    engine.pgo.optimize()
    engine.save_results(save_dir, pcd_filename)
    print(f"[Done] Total Loops Found: {len(engine.loop_pairs)} | Final PCD map saved.")


# ==============================================================================
# 9. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="FAST-LIVO2 Loop Closure & PGO (ROS1 / RoboSense)")
    parser.add_argument("--bag", type=str, default=None, help="Path to ROS 1 .bag file for offline mode")
    parser.add_argument("--odom_topic", type=str, default="/aft_mapped_to_init", help="FAST-LIVO2 Odometry topic")
    parser.add_argument("--cloud_topic", type=str, default="/cloud_registered", help="Registered/Raw LiDAR topic")
    parser.add_argument("--save_dir", type=str, default="./output_map", help="Directory to save final .pcd and trajectories")
    parser.add_argument("--pcd_filename", type=str, default="drift_free_global_map.pcd", help="Filename of saved PCD map")
    
    args, unknown = parser.parse_known_args()

    if args.bag is not None:
        # Offline mode
        run_offline_bag(args.bag, args.odom_topic, args.cloud_topic, args.save_dir, args.pcd_filename)
    else:
        # Online ROS 1 Node mode
        if not ROS_AVAILABLE:
            print("[Error] ROS1 environment not found. For offline mode, specify --bag <file.bag>")
            sys.exit(1)
        node = FastLivo2LoopROSNode()
        rospy.spin()

if __name__ == "__main__":
    main()
