<div align="center">
  <h1>FAST-LIVO2</h1>
  <h3>Fast, Direct LiDAR-Inertial-Visual Odometry</h3>
  
  [![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/gpl-2.0)
  [![ROS](https://img.shields.io/badge/ROS-Supported-brightgreen.svg)](http://wiki.ros.org/ROS/Installation)
  [![Paper](https://img.shields.io/badge/Arxiv-2408.14035-red)](https://arxiv.org/abs/2408.14035)

</div>

<p align="center">
  <b>FAST-LIVO2</b> is an efficient, robust, and highly accurate LiDAR-inertial-visual (LIV) fusion localization and mapping system. It significantly advances real-time 3D reconstruction and onboard robotic localization, especially in challenging, severely degraded environments.
</p>

---

## 📑 Table of Contents

- [📢 News](#-news)
- [✨ Key Features](#-key-features)
- [📚 Related Resources](#-related-resources)
- [⚙️ Prerequisites](#️-prerequisites)
- [🚀 Build Instructions](#-build-instructions)
- [🎮 Running Examples](#-running-examples)
- [📄 License & Contact](#-license--contact)

---

## 📢 News

- 🔓 **2025-01-23**: Source code officially released!
- 🎉 **2024-10-01**: Accepted by **T-RO '24**!
- 🚀 **2024-07-02**: Conditionally accepted for publication.

---

## ✨ Key Features

- **Direct LiDAR-Visual Fusion:** Avoids explicit feature extraction, leveraging raw LiDAR points and direct photometric errors to reduce computational latency.
- **High Efficiency:** Built on an Error-State Iterated Kalman Filter (ESIKF) allowing sequential updates that elegantly resolve sensor dimension mismatch.
- **Unified Mapping:** Employs a single voxel map that inherently fuses geometric and visual data, maintaining high precision mapping.
- **Robustness:** Ensures highly stable pose estimation even in visually degraded or geometrically featureless environments.

<div align="center">
    <img src="pics/Framework.png" alt="FAST-LIVO2 Framework" width="90%">
</div>

---

## 📚 Related Resources

### 📝 Papers
- [**FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry**](https://arxiv.org/pdf/2408.14035)
- [**FAST-LIVO2 on Resource-Constrained Platforms**](https://arxiv.org/pdf/2501.13876)
- [**FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry**](https://arxiv.org/pdf/2203.00893)
- [**FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second**](https://arxiv.org/pdf/2507.17210)

### 📹 Media
Watch our accompanying demonstrations on [**YouTube**](https://youtu.be/6dF2DzgbtlY) or [**Bilibili**](https://www.bilibili.com/video/BV1Ezxge7EEi).

### 🛠 Tools & Datasets
- **Hardware:** We have open-sourced our handheld device (CAD, STM32 code, wiring, and drivers). Find it here: [LIV_handhold](https://github.com/xuankuzcr/LIV_handhold).
- **Dataset:** The [FAST-LIVO2-Dataset](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/ErdFNQtjMxZOorYKDTtK4ugBkogXfq1OfDm90GECouuIQA?e=KngY9Z) used for our evaluations is available online.
- **Calibration:** We highly recommend using the [FAST-Calib](https://github.com/hku-mars/FAST-Calib) toolkit to obtain extrinsic parameters that can be seamlessly incorporated into FAST-LIVO2 YAML configs.

---

## ⚙️ Prerequisites

### 1. Ubuntu and ROS
- Supported OS: Ubuntu 18.04, 20.04.
- Follow the standard [ROS Installation Guide](http://wiki.ros.org/ROS/Installation).

### 2. Core Dependencies
- **PCL** (>= 1.8): [Installation Instructions](https://pointclouds.org/)
- **Eigen** (>= 3.3.4): [Installation Instructions](https://eigen.tuxfamily.org/)
- **OpenCV** (>= 4.2): [Installation Instructions](http://opencv.org/)

### 3. Sophus
Install the non-templated/double-only version of Sophus:
```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build && cmake ..
make
sudo make install
```

### 4. Vikit
Vikit is necessary for camera models and interpolation routines. Clone it into your catkin workspace `src` directory (Note: it is different from the version used in FAST-LIVO1):
```bash
cd ~/catkin_ws/src
git clone https://github.com/xuankuzcr/rpg_vikit.git
```

---

## 🚀 Build Instructions

Clone this repository directly into your catkin workspace and build:

```bash
cd ~/catkin_ws/src
# Clone the repository
git clone https://github.com/BavanPrabahar/FAST-LIVO2.git
cd ../
# Build the workspace
catkin_make
# Source the setup script
source ~/catkin_ws/devel/setup.bash
```

---

## 🎮 Running Examples

1. **Download Dataset:** Download the dataset from [Global-LVBA Section IV](https://github.com/xuankuzcr/Global-LVBA).
2. **Launch Node:**
   ```bash
   roslaunch fast_livo mapping_avia.launch
   ```
3. **Play Rosbag:**
   ```bash
   rosbag play YOUR_DOWNLOADED.bag
   ```

---

## 📄 License & Contact

### License
This package is released under the **[GPLv2 License](http://www.gnu.org/licenses/)**. For commercial licensing, please reach out to the original developers.

### Original Authors
FAST-LIVO2 is primarily developed by **Chunran Zheng** ([@xuankuzcr](https://github.com/xuankuzcr)).
- Inquiries: [zhengcr@connect.hku.hk](mailto:zhengcr@connect.hku.hk)
- Prof. Fu Zhang: [fuzhang@hku.hk](mailto:fuzhang@hku.hk)

---
*Maintained and enhanced by [BavanPrabahar](https://github.com/BavanPrabahar).*
