#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Float32MultiArray, String

class PControllerNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(PControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        
        # Get vehicle name
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribe to camera image
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.camera_sub = rospy.Subscriber(
            self.camera_topic,
            CompressedImage,
            self.camera_callback,
            queue_size=1,
            buff_size=10000000
        )
        
        # Publishers
        # 1. Yellow line detection visualization
        self.yellow_line_pub = rospy.Publisher(
            f"/{self.vehicle_name}/yellow_line/image/compressed",
            CompressedImage,
            queue_size=1
        )
        
        # 2. White line detection visualization
        self.white_line_pub = rospy.Publisher(
            f"/{self.vehicle_name}/white_line/image/compressed",
            CompressedImage,
            queue_size=1
        )
        
        # 3. Command velocity publisher
        self.cmd_vel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        
        # 4. Target path visualization
        self.target_path_pub = rospy.Publisher(
            f"/{self.vehicle_name}/target_path",
            Float32MultiArray,
            queue_size=1
        )

        # 5. Controller state publisher (for debugging)
        self.controller_state_pub = rospy.Publisher(
            f"/{self.vehicle_name}/controller_state",
            String,
            queue_size=1
        )
        
        # Initialize controller parameters
        self.kp = 0.035  # Proportional gain
        
        # Controller state
        self.error = 0
        
        # Forward velocity (constant for straight line test)
        self.forward_velocity = 0.35  # m/s
        
        # Lane detection parameters
        # HSV thresholds for yellow lane
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        
        # HSV thresholds for white lane
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        
        # Image parameters
        self.img_width = 640
        self.img_height = 480
        self.roi_top = 250  # Region of interest: top pixel row
        
        # Timer for controller update (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        self.log("P Controller node initialized")
    
    def camera_callback(self, msg):
        """Process the camera image to detect lanes"""
        try:
            # Convert compressed image to CV image
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
            
            # Crop to region of interest (lower part of the image)
            roi = img[self.roi_top:, :]
            
            # Detect yellow and white lanes
            yellow_mask, yellow_center, yellow_contour = self.detect_lane(roi, 'yellow')
            white_mask, white_center, white_contour = self.detect_lane(roi, 'white')
            
            # Calculate error based on lane detection
            self.calculate_error(yellow_center, white_center, roi.shape[1])
            
            # Visualize lane detection if subscribers exist
            if self.yellow_line_pub.get_num_connections() > 0:
                self.publish_lane_visualization(roi, yellow_mask, yellow_contour, 'yellow')
            
            if self.white_line_pub.get_num_connections() > 0:
                self.publish_lane_visualization(roi, white_mask, white_contour, 'white')
                
        except Exception as e:
            self.logerr(f"Error processing camera image: {str(e)}")
    
    def detect_lane(self, img, color):
        """Detect lane of specified color in the image"""
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Apply color mask
        if color == 'yellow':
            mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        else:  # white
            mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize center position
        center_x = None
        largest_contour = None
        
        # Find the largest contour (most likely the lane)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only process if contour is large enough
            if cv2.contourArea(largest_contour) > 50:
                # Calculate moments to find centroid
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
        
        return mask, center_x, largest_contour
    
    def publish_lane_visualization(self, img, mask, contour, color):
        """Publish visualization of lane detection"""
        # Create a color visualization image
        vis_img = img.copy()
        
        # Overlay the mask in color
        color_mask = np.zeros_like(vis_img)
        if color == 'yellow':
            color_mask[mask > 0] = [0, 255, 255]  # Yellow color
        else:
            color_mask[mask > 0] = [255, 255, 255]  # White color
        
        # Add mask to image with transparency
        vis_img = cv2.addWeighted(vis_img, 1, color_mask, 0.5, 0)
        
        # Draw contour if available
        if contour is not None:
            cv2.drawContours(vis_img, [contour], -1, (0, 0, 255), 2)
        
        # Convert back to compressed image and publish
        msg = self.bridge.cv2_to_compressed_imgmsg(vis_img)
        if color == 'yellow':
            self.yellow_line_pub.publish(msg)
        else:
            self.white_line_pub.publish(msg)
    
    def calculate_error(self, yellow_center, white_center, img_width):
        """Calculate the error for lane following based on detected lanes"""
        # Center of the image
        img_center = img_width // 2
        
        if yellow_center is not None and white_center is not None:
            # If both lanes are detected, follow the center
            lane_center = (yellow_center + white_center) // 2
            self.error = lane_center - img_center
        elif yellow_center is not None:
            # If only yellow lane is detected, stay a fixed distance to the right
            self.error = yellow_center - (img_center - 100)  # Offset to stay right of yellow line
        elif white_center is not None:
            # If only white lane is detected, stay a fixed distance to the left
            self.error = white_center - (img_center + 100)  # Offset to stay left of white line
        # If no lanes detected, keep the previous error (no update)
    
    def control_loop(self, event):
        """P control loop for lane following"""
        # Skip if no error has been calculated yet (e.g., no lanes detected yet)
        if self.error is None:
            return
        
        # P controller: omega = -kp * error
        omega = -self.kp * self.error
        
        # Publish controller state for debugging
        controller_state = f"P - Error: {self.error:.2f}, Output: {omega:.4f}"
        self.controller_state_pub.publish(controller_state)
        
        # Create and publish velocity command
        cmd = Twist2DStamped()
        cmd.v = self.forward_velocity
        cmd.omega = omega
        self.cmd_vel_pub.publish(cmd)
    
    def stop(self):
        """Stop the robot by publishing zero velocity"""
        cmd = Twist2DStamped()
        cmd.v = 0
        cmd.omega = 0
        self.cmd_vel_pub.publish(cmd)

if __name__ == '__main__':
    # Initialize the node
    controller_node = PControllerNode(node_name='p_controller_node')
    
    # Spin forever
    rospy.spin()
    
    # Make sure to stop the robot when the node is shut down
    controller_node.stop()
