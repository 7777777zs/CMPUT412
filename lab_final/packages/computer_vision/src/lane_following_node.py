#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
from std_msgs.msg import Float32MultiArray, String
import time

class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LaneFollowingNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        
        # Get vehicle name
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize controller parameters (PID is the default for full lap)
        self.kp = 0.035  # Proportional gain
        self.kd = 0.01   # Derivative gain
        self.ki = 0.001  # Integral gain
        
        # Controller state
        self.error = 0
        self.last_error = 0
        self.integral = 0
        self.derivative = 0
        self.last_time = rospy.get_time()
        
        # Lane tracking parameters
        self.lane_width_pixels = 200  # Approximate lane width in pixels
        
        # Speed parameters
        self.normal_speed = 0.4      # Regular forward speed
        self.turn_speed = 0.3        # Speed during sharp turns
        self.current_speed = self.normal_speed
        
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
        # 1. Visualization publisher
        self.vis_pub = rospy.Publisher(
            f"/{self.vehicle_name}/lane_following/image/compressed",
            CompressedImage,
            queue_size=1
        )
        
        # 2. Command velocity publisher
        self.cmd_vel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        
        # 3. Controller status publisher
        self.status_pub = rospy.Publisher(
            f"/{self.vehicle_name}/lane_following/status",
            String,
            queue_size=1
        )
        
        # 4. LED pattern publisher
        self.led_pattern_pub = rospy.Publisher(
            f"/{self.vehicle_name}/led_emitter_node/led_pattern",
            LEDPattern,
            queue_size=1
        )
        
        # Lane detection parameters
        # HSV thresholds for yellow lane
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        
        # HSV thresholds for white lane
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 60, 255])
        
        # Image parameters
        self.img_width = 640
        self.img_height = 480
        
        # Turn parameters for intersections
        self.is_turning = False
        self.turn_start_time = 0
        self.turn_duration = 0
        
        # Region of interest (ROI) - adjustable based on camera position
        self.roi_top = 250  # Top row of ROI
        self.roi_height = 120  # Height of ROI
        self.roi_width = 640  # Width of ROI (full width)
        
        # Timer for controller update (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        self.log("Lane following node initialized with PID control")
    
    def camera_callback(self, msg):
        """Process the camera image to detect lanes"""
        try:
            # Convert compressed image to CV image
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
            
            # Crop to region of interest (lower part of the image)
            roi = img[self.roi_top:self.roi_top + self.roi_height, 0:self.roi_width]
            
            # Make a copy for visualization
            vis_img = roi.copy()
            
            # Detect lanes
            yellow_mask, yellow_center, yellow_contour = self.detect_lane(roi, 'yellow')
            white_mask, white_center, white_contour = self.detect_lane(roi, 'white')
            
            # Visualize lane detection
            if yellow_contour is not None:
                cv2.drawContours(vis_img, [yellow_contour], -1, (0, 255, 255), 2)
                if yellow_center is not None:
                    cv2.circle(vis_img, (yellow_center, roi.shape[0]//2), 5, (0, 255, 255), -1)
            
            if white_contour is not None:
                cv2.drawContours(vis_img, [white_contour], -1, (255, 255, 255), 2)
                if white_center is not None:
                    cv2.circle(vis_img, (white_center, roi.shape[0]//2), 5, (255, 255, 255), -1)
            
            # Calculate error based on lane detection
            self.calculate_error(yellow_center, white_center, roi.shape[1])
            
            # Draw error and centerline visualizations
            self.draw_error_visualization(vis_img)
            
            # Publish visualization
            if self.vis_pub.get_num_connections() > 0:
                vis_msg = self.bridge.cv2_to_compressed_imgmsg(vis_img)
                self.vis_pub.publish(vis_msg)
                
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
            # Filter contours by area to eliminate noise
            valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Calculate moments to find centroid
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
        
        return mask, center_x, largest_contour
    
    def draw_error_visualization(self, img):
        """Add visual indicators for error and target path"""
        h, w = img.shape[:2]
        
        # Draw center of image (reference)
        center_x = w // 2
        cv2.line(img, (center_x, 0), (center_x, h), (0, 0, 255), 1)
        
        # Draw error line
        if self.error is not None:
            # Calculate position based on error
            error_pos = center_x + self.error
            if 0 <= error_pos < w:
                cv2.line(img, (error_pos, 0), (error_pos, h), (255, 0, 0), 2)
                
                # Draw a line connecting center to error position
                cv2.line(img, (center_x, h//2), (error_pos, h//2), (0, 255, 0), 2)
    
    def calculate_error(self, yellow_center, white_center, img_width):
        """Calculate the error for lane following based on detected lanes"""
        # Center of the image
        img_center = img_width // 2
        
        # Check if we have both lane markings
        if yellow_center is not None and white_center is not None:
            # Both lanes detected - aim for the center
            lane_center = (yellow_center + white_center) // 2
            self.error = lane_center - img_center
            
            # Adjust speed based on lane width
            lane_width = abs(white_center - yellow_center)
            if lane_width < self.lane_width_pixels * 0.7:  # Lane is narrower than expected
                self.current_speed = self.turn_speed  # Slow down for turns
            else:
                self.current_speed = self.normal_speed  # Regular speed
            
        elif yellow_center is not None:
            # Only yellow lane detected - stay a fixed distance to the right
            self.error = yellow_center - (img_center - 100)  # Offset by approx lane width/2
            self.current_speed = self.turn_speed  # Slow down when only one lane is visible
            
        elif white_center is not None:
            # Only white lane detected - stay a fixed distance to the left
            self.error = white_center - (img_center + 100)  # Offset by approx lane width/2
            self.current_speed = self.turn_speed  # Slow down when only one lane is visible
            
        # If no lanes detected, keep the previous error (no update)
    
    def control_loop(self, event):
        """PID control loop for lane following"""
        if self.error is None:
            return
        
        # Calculate dt
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        if dt == 0:
            return
        self.last_time = current_time
        
        # Calculate derivative
        self.derivative = (self.error - self.last_error) / dt
        
        # Calculate integral with anti-windup
        self.integral += self.error * dt
        
        # Anti-windup: Limit integral term
        if self.integral > 100:
            self.integral = 100
        elif self.integral < -100:
            self.integral = -100
        
        # PID controller for omega
        omega = -self.kp * self.error - self.kd * self.derivative - self.ki * self.integral
        
        # Limit the maximum omega to prevent excessive rotation
        max_omega = 8.0
        omega = max(min(omega, max_omega), -max_omega)
        
        # Create and publish velocity command
        cmd = Twist2DStamped()
        cmd.v = self.current_speed  # Use adaptive speed
        cmd.omega = omega
        self.cmd_vel_pub.publish(cmd)
        
        # Save error for next iteration
        self.last_error = self.error
        
        # Publish status for debugging/monitoring
        status_msg = (f"Error: {self.error:.2f}, D: {self.derivative:.2f}, "
                     f"I: {self.integral:.2f}, Omega: {omega:.2f}, Speed: {self.current_speed:.2f}")
        self.status_pub.publish(status_msg)
    
    def stop(self):
        """Stop the robot by publishing zero velocity"""
        cmd = Twist2DStamped()
        cmd.v = 0
        cmd.omega = 0
        self.cmd_vel_pub.publish(cmd)
        self.log("Lane following node stopped")
    
    def set_led_pattern(self, rgb_values=[]):
        """Set LED pattern, useful for indicating state or debugging"""
        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()
        
        # Default: all LEDs off
        if not rgb_values:
            rgb_values = [(0, 0, 0)] * 5
            
        # Set colors for each LED
        for i, rgb in enumerate(rgb_values):
            color = [0, 0, 0, 0, 0]  # R, G, B, W, NOTHING
            color[0] = rgb[0]  # R
            color[1] = rgb[1]  # G
            color[2] = rgb[2]  # B
            
            for j in range(len(color)):
                pattern_msg.rgb_vals.append(color[j])
        
        self.led_pattern_pub.publish(pattern_msg)

if __name__ == '__main__':
    # Initialize the node
    lane_following_node = LaneFollowingNode(node_name='lane_following_node')
    
    try:
        # Spin until interrupted
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to stop the robot when the node is shut down
        lane_following_node.stop()