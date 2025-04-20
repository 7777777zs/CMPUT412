#!/usr/bin/env python3

import rospy
from os import uname
from re import match
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
import cv2
from cv_bridge import CvBridge


class CameraDistortionNode(DTROS):
    def __init__(self, node_name):
        # Initialize DTROS parent class
        super(CameraDistortionNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        # Static parameters
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._custom_topic = f"/{self._vehicle_name}/undistorted_node/image/compressed"

        self._homo = np.array([
            [-4.4176150901508024e-05,  0.0004846965281425196,  0.30386285736827856],
            [-0.0015921548874079563,   1.0986376221035314e-05, 0.4623061006515125],
            [-0.00029009271219456394,  0.011531091762722323,  -1.4589875279686733]
        ])

        # Subscribe to camera info
        self.camera_info_sub = rospy.Subscriber(
            f'/{self._vehicle_name}/camera_node/camera_info', CameraInfo, self.camera_info_callback
        )

        self.D = None  # Storage for distortion coefficients
        self.K = None  # Storage for intrinsic matrix
        self.height = None
        self.width = None
        self.map1 = None  # Precomputed remap matrices
        self.map2 = None

        self.scale_factor = 0.5  # Reduce resolution to 50%

        # Bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # Construct subscriber
        self.sub = rospy.Subscriber(
            self._camera_topic, CompressedImage, self.camera_reader_callback, queue_size=5)
        self.pub = rospy.Publisher(
            self._custom_topic, CompressedImage, queue_size=5)

        # Construct the scaling matrix
        S = np.array([
            [self.scale_factor, 0, 0],
            [0, self.scale_factor, 0],
            [0, 0, 1]  # Keep the last row unchanged
        ])

        # Scale the homography matrix
        self._homo = S @ self._homo @ np.linalg.inv(S)

        rospy.set_param(
            f"/{self._vehicle_name}/updated_extrinsics", self._homo.flatten().tolist())

    def camera_reader_callback(self, msg):
        if rospy.is_shutdown():
            return

        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        if self.K is None or self.D is None:
            rospy.logwarn("Camera parameters not received yet.")
            return

        # Compute remap matrices once
        if self.map1 is None or self.map2 is None:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.K, (self.width, self.height), cv2.CV_16SC2)

        # Apply undistortion using precomputed maps
        dst = cv2.remap(image, self.map1, self.map2,
                        interpolation=cv2.INTER_LINEAR)

        # height, _ = dst.shape[:2]

        # # Crop bottom 50% (remove top half)
        # dst = dst[height // 2:, :]

        # Get image dimensions
        # height, width = dst.shape[:2]

        # # Create a black bar for the top third
        # black_bar = np.zeros((height // 3, width, 3), dtype=np.uint8)

        # # Replace the top third with the black bar
        # dst[:height // 3, :] = black_bar

        # dst = cv2.resize(dst, None, fx=self.scale_factor, fy=self.scale_factor,
        #                  interpolation=cv2.INTER_AREA)

        # blur = cv2.GaussianBlur(dst, (5, 5), 0)

        # Convert back to compressed image message
        # image_message = self._bridge.cv2_to_compressed_imgmsg(blur)

        image_message = self._bridge.cv2_to_compressed_imgmsg(dst)

        # Publish the compressed image
        self.pub.publish(image_message)

    def camera_info_callback(self, msg):
        """Callback to receive and store distortion coefficients."""
        self.D = np.array(msg.D)
        self.K = np.array(msg.K).reshape(3, 3)
        self.width = msg.width
        self.height = msg.height

        # Destroy the subscription after receiving the values once
        self.camera_info_sub.unregister()
        rospy.loginfo("Camera info subscription closed.")


if __name__ == '__main__':
    # Create the node
    node = CameraDistortionNode(node_name='camera_reader_node')
    # Keep spinning
    rospy.spin()
