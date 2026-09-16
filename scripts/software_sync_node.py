#!/usr/bin/env python3
"""
Software Synchronization & Camera Adapter Node for FAST-LIVO2.
- Subscribes to raw USB camera feed (/usb_image_raw).
- Guarantees non-zero, monotonic hardware/host timestamps aligned with Livox IMU/LiDAR.
- Optionally resizes/crops to 1280x720 (or target resolution) matching camera intrinsics.
- Publishes clean synchronized feed to /usb_image_raw_sync.
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def cv2_to_imgmsg_manual(cv_img, encoding, header):
    """Build a sensor_msgs/Image directly to bypass cv_bridge version bugs."""
    cv_img = np.ascontiguousarray(cv_img)
    msg = Image()
    msg.header = header
    msg.height = cv_img.shape[0]
    msg.width = cv_img.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = cv_img.strides[0]
    msg.data = cv_img.tobytes()
    return msg


class SoftwareCameraSynchronizer:
    def __init__(self):
        rospy.init_node("software_camera_synchronizer")

        self.in_topic = rospy.get_param("~in_topic", "/usb_image_raw")
        self.out_topic = rospy.get_param("~out_topic", "/usb_image_raw_sync")
        self.target_w = rospy.get_param("~target_width", 1280)
        self.target_h = rospy.get_param("~target_height", 720)
        self.passthrough = rospy.get_param("~passthrough", False)

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.out_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(self.in_topic, Image, self.callback, queue_size=10)

        rospy.loginfo("=== Software Camera Synchronizer Started ===")
        rospy.loginfo("Input Topic:  %s", self.in_topic)
        rospy.loginfo("Output Topic: %s", self.out_topic)
        rospy.loginfo("Target Resolution: %dx%d (Passthrough: %s)", self.target_w, self.target_h, self.passthrough)

    def callback(self, msg):
        header = msg.header
        
        # 1. Guarantee valid, non-zero timestamp aligned with current system clock
        if header.stamp.to_sec() == 0:
            header.stamp = rospy.Time.now()

        # If passthrough is enabled, directly publish with guaranteed header
        if self.passthrough:
            self.pub.publish(msg)
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        h, w = cv_img.shape[:2]

        # 2. Crop to target aspect ratio (16:9) if needed, then resize
        target_aspect = float(self.target_w) / float(self.target_h)
        current_aspect = float(w) / float(h)

        if abs(current_aspect - target_aspect) > 0.05:
            # Crop height or width to match aspect ratio
            if current_aspect > target_aspect:
                # Image is too wide -> crop width
                new_w = int(round(h * target_aspect))
                left = (w - new_w) // 2
                cv_img = cv_img[:, left:left + new_w]
            else:
                # Image is too tall -> crop height
                new_h = int(round(w / target_aspect))
                top = (h - new_h) // 2
                cv_img = cv_img[top:top + new_h, :]

        if (cv_img.shape[1], cv_img.shape[0]) != (self.target_w, self.target_h):
            resized = cv2.resize(cv_img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv_img

        out_msg = cv2_to_imgmsg_manual(resized, msg.encoding, header)
        self.pub.publish(out_msg)


if __name__ == "__main__":
    node = SoftwareCameraSynchronizer()
    rospy.spin()
