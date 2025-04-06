#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from duckietown.dtros import DTROS, NodeType

class CameraProcessingNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(CameraProcessingNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        
        # Get vehicle name from environment variable
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Camera info and image topics
        self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"
        self.distorted_img_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        
        # Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False
        
        # Image processing parameters
        self.resize_width = 320  # Set your desired width
        self.resize_height = 240  # Set your desired height
        self.blur_kernel_size = (5, 5)  # Gaussian blur kernel size
        self.blur_sigma = 0  # 0 means auto-calculate based on kernel size
        
        # Create publishers for processed images
        self.undistorted_pub = rospy.Publisher(
            f"/{self.vehicle_name}/camera_node/image/undistorted",
            Image,
            queue_size=1
        )
        
        self.processed_pub = rospy.Publisher(
            f"/{self.vehicle_name}/camera_node/image/processed",
            Image,
            queue_size=1
        )
        
        # Subscribe to camera info topic
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback
        )
        
        # Subscribe to compressed image topic
        self.image_sub = rospy.Subscriber(
            self.distorted_img_topic,
            CompressedImage,
            self.image_callback
        )
        
        # Set processing rate to 5 Hz as suggested in the exercise
        self.rate = rospy.Rate(5)
        
        self.log("Camera processing node initialized")
    
    def camera_info_callback(self, msg):
        """Extract camera intrinsic parameters from camera_info topic"""
        if not self.got_camera_info:
            # Extract the camera matrix (K) from the message
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            
            # Extract distortion coefficients (D) from the message
            self.dist_coeffs = np.array(msg.D)
            
            self.got_camera_info = True
            self.log(f"Received camera intrinsic parameters:")
            self.log(f"Camera Matrix (K): \n{self.camera_matrix}")
            self.log(f"Distortion Coefficients (D): {self.dist_coeffs}")
    
    def resize_image(self, image):
        """Resize image to the specified dimensions"""
        return cv2.resize(image, (self.resize_width, self.resize_height))
    
    def apply_blur(self, image):
        """Apply Gaussian blur to the image"""
        return cv2.GaussianBlur(image, self.blur_kernel_size, self.blur_sigma)
    
    def image_callback(self, msg):
        """Process incoming compressed images, undistort them, and publish the result"""
        # Skip if we haven't received camera parameters yet
        if not self.got_camera_info:
            return
        
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            distorted_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Get image dimensions
            h, w = distorted_img.shape[:2]
            
            # Calculate optimal camera matrix
            # Alpha parameter (0-1): 0 = all distorted pixels visible, 1 = all pixels visible
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h)
            )
            
            # Manual undistortion using cv2.undistort
            undistorted_img = cv2.undistort(
                distorted_img, 
                self.camera_matrix, 
                self.dist_coeffs, 
                None, 
                new_camera_matrix
            )
            
            # Optional: Crop the image to the ROI
            x, y, w, h = roi
            if all([x, y, w, h]):  # Make sure ROI is valid
                undistorted_img = undistorted_img[y:y+h, x:x+w]
            
            # Convert to ROS format and publish undistorted image
            undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_img, "bgr8")
            undistorted_msg.header = msg.header  # Keep original timestamp
            self.undistorted_pub.publish(undistorted_msg)
            
            # Apply pre-processing: resize the image
            resized_img = self.resize_image(undistorted_img)
            
            # Apply pre-processing: blur the image
            processed_img = self.apply_blur(resized_img)
            
            # Convert to ROS format and publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_img, "bgr8")
            processed_msg.header = msg.header  # Keep original timestamp
            self.processed_pub.publish(processed_msg)
            
        except Exception as e:
            self.logerr(f"Error processing image: {e}")
    
    def run(self):
        """Main loop to keep the node running at the specified rate"""
        self.log("Starting camera processing node")
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    # Initialize the node
    camera_node = CameraProcessingNode(node_name='camera_processing_node')
    # Run the node
    camera_node.run()
