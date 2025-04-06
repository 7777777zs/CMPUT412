#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rospy
import time
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown_msgs.msg import Twist2DStamped, LEDPattern, WheelsCmdStamped
from std_msgs.msg import Float32MultiArray, String, ColorRGBA

class ComputerVisionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(ComputerVisionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        
        # Get vehicle name
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # State machine states
        self.STATE_SEARCHING = "searching"         # Looking for colored lines
        self.STATE_APPROACHING = "approaching"     # Approaching a detected line
        self.STATE_STOPPING = "stopping"           # Stopping at the line
        self.STATE_TURNING_RIGHT = "turning_right" # Executing right turn (blue)
        self.STATE_TURNING_LEFT = "turning_left"   # Executing left turn (green)
        self.STATE_MOVING_STRAIGHT = "moving_straight" # Moving straight (red)
        self.STATE_CONTINUE = "continue"           # Continue forward
        
        # Current state
        self.state = self.STATE_SEARCHING
        self.state_start_time = rospy.get_time()
        
        # Store detected color
        self.detected_color = None
        self.detected_distance = 0
        self.approach_speed = 0.15  # Slower speed when approaching line
        self.normal_speed = 0.22    # Normal forward speed
        self.turn_speed = 0.15      # Speed during turns
        
        # Timing parameters
        self.stop_duration = 4.0    # Time to stop at line (3-5 seconds)
        self.right_turn_duration = 2.5  # Time for 90-degree right turn
        self.left_turn_duration = 2.5   # Time for 90-degree left turn
        self.straight_distance = 0.3    # Distance to move straight (30+ cm)
        
        # Distance thresholds
        self.detection_distance = 30.0  # Minimum detection distance (cm)
        self.stopping_distance = 10.0   # Distance to stop from line (cm)
        
        # Camera calibration matrix
        self.K = None
        self.D = None
        self.camera_info_received = False
        
        # Subscribe to camera info for distortion parameters
        self.camera_info_sub = rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/camera_info",
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )
        
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
        # 1. Undistorted image publisher
        self.undistorted_pub = rospy.Publisher(
            f"/{self.vehicle_name}/camera_node/image/undistorted",
            CompressedImage,
            queue_size=1
        )
        
        # 2. Color detection visualization
        self.color_detect_pub = rospy.Publisher(
            f"/{self.vehicle_name}/color_detection/image/compressed",
            CompressedImage,
            queue_size=1
        )
        
        # 3. Command velocity publisher
        self.cmd_vel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        
        # 4. LED pattern publisher
        self.led_pattern_pub = rospy.Publisher(
            f"/{self.vehicle_name}/led_emitter_node/led_pattern",
            LEDPattern,
            queue_size=1
        )
        
        # 5. State publisher
        self.state_pub = rospy.Publisher(
            f"/{self.vehicle_name}/computer_vision_node/state",
            String,
            queue_size=1
        )
        
        # HSV color thresholds
        # These are starter values - might need tuning
        self.blue_lower = np.array([100, 50, 50])
        self.blue_upper = np.array([130, 255, 255])
        
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])
        
        # Image parameters
        self.img_width = 640
        self.img_height = 480
        
        # Region of interest (ROI)
        self.roi_top = 300  # Top row of ROI (lower in the image)
        self.roi_height = 180  # Height of ROI
        
        # Timer for state machine update (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.state_machine_loop)
        
        self.log("Computer vision node initialized with colored line behaviors")
    
    def camera_info_callback(self, msg):
        """Extract camera calibration parameters"""
        if not self.camera_info_received:
            # Get camera matrix and distortion coefficients
            self.K = np.array(msg.K).reshape((3, 3))
            self.D = np.array(msg.D)
            self.camera_info_received = True
            self.log(f"Camera calibration received. K={self.K}, D={self.D}")
    
    def camera_callback(self, msg):
        """Process the camera image for lane detection and navigation"""
        try:
            # Convert compressed image to CV image
            distorted_img = self.bridge.compressed_imgmsg_to_cv2(msg)
            
            # Skip if camera info not received yet
            if not self.camera_info_received:
                return
            
            # 1. Undistort the image using camera intrinsics
            undistorted_img = self.undistort_image(distorted_img)
            
            # 2. Image pre-processing
            processed_img = self.preprocess_image(undistorted_img)
            
            # 3. Color detection
            detected_color, contour_rect, contour_area, distance = self.detect_color_lines(processed_img)
            
            # Store detection results
            if contour_area > 500:  # Minimum area threshold
                self.detected_color = detected_color
                self.detected_distance = distance
            
            # Visualize detection if subscribers exist
            if self.color_detect_pub.get_num_connections() > 0:
                vis_img = self.visualize_detection(undistorted_img, detected_color, contour_rect)
                vis_msg = self.bridge.cv2_to_compressed_imgmsg(vis_img)
                self.color_detect_pub.publish(vis_msg)
            
            # Publish undistorted image
            if self.undistorted_pub.get_num_connections() > 0:
                undistorted_msg = self.bridge.cv2_to_compressed_imgmsg(undistorted_img)
                self.undistorted_pub.publish(undistorted_msg)
                
        except Exception as e:
            self.logerr(f"Error processing camera image: {str(e)}")
    
    def undistort_image(self, img):
        """Undistort image using camera calibration parameters"""
        h, w = img.shape[:2]
        
        # Get optimal new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
        
        # Undistort
        dst = cv2.undistort(img, self.K, self.D, None, newcameramtx)
        
        # Crop the image
        x, y, w, h = roi
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def preprocess_image(self, img):
        """Preprocess image: resize, convert to HSV, blur, etc."""
        # Extract region of interest
        roi = img[self.roi_top:self.roi_top + self.roi_height, :]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def detect_color_lines(self, hsv_img):
        """Detect blue, red, and green lines in the image"""
        # Initialize variables
        detected_color = None
        max_area = 0
        max_contour_rect = None
        distance = 0
        
        # Process each color
        for color_name, color_data in [
            ("blue", (self.blue_lower, self.blue_upper)),
            ("red", (self.red_lower1, self.red_upper1, self.red_lower2, self.red_upper2)),
            ("green", (self.green_lower, self.green_upper))
        ]:
            # Create a mask for color detection
            if color_name == "red":
                # Red requires two ranges in HSV
                mask1 = cv2.inRange(hsv_img, color_data[0], color_data[1])
                mask2 = cv2.inRange(hsv_img, color_data[2], color_data[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_img, color_data[0], color_data[1])
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour for this color
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # If this is the largest contour of all colors, update detected color
                if area > max_area and area > 500:  # Minimum area threshold
                    max_area = area
                    detected_color = color_name
                    
                    # Get bounding rectangle for the contour
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    max_contour_rect = (x, y, w, h)
                    
                    # Estimate distance based on the position of the contour (y position)
                    # Lower y means further away (top of the image)
                    # This is a simple approximation - would need calibration for accurate distance
                    img_height = hsv_img.shape[0]
                    relative_position = (img_height - (y + h)) / img_height
                    distance = relative_position * 100  # Convert to approximate cm
        
        return detected_color, max_contour_rect, max_area, distance
    
    def visualize_detection(self, img, detected_color, contour_rect):
        """Create visualization of color detection"""
        vis_img = img.copy()
        
        # Draw ROI region
        cv2.rectangle(vis_img, (0, self.roi_top), 
                     (vis_img.shape[1], self.roi_top + self.roi_height), (255, 0, 0), 2)
        
        # Draw state and color detection
        cv2.putText(vis_img, f"State: {self.state}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if detected_color:
            # Get color for visualization
            color_rgb = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
            color_text = detected_color.upper()
            
            # Add label with detected color
            cv2.putText(vis_img, f"Detected: {color_text}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rgb[detected_color], 2)
            
            # Draw bounding box if available
            if contour_rect:
                x, y, w, h = contour_rect
                # Adjust y to account for ROI position
                y += self.roi_top
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), color_rgb[detected_color], 2)
                
                # Add distance estimate
                cv2.putText(vis_img, f"Distance: {self.detected_distance:.1f} cm", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_img
    
    def set_led_pattern(self, pattern):
        """Set LED pattern for the robot (left or right signals, or off)"""
        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()
        
        # Pattern definitions
        if pattern == "off":
            # All LEDs off
            colors = [(0, 0, 0)] * 5
        elif pattern == "left":
            # Left turn signal (leftmost LEDs in yellow)
            colors = [(1, 1, 0), (1, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        elif pattern == "right":
            # Right turn signal (rightmost LEDs in yellow)
            colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0), (1, 1, 0)]
        elif pattern == "brake":
            # Brake lights (all LEDs in red)
            colors = [(1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)]
        else:
            # Default: all off
            colors = [(0, 0, 0)] * 5
        
        # Set colors for each LED using ColorRGBA objects
        for r, g, b in colors:
            # Create ColorRGBA object for each LED
            color = ColorRGBA()
            color.r = float(r)
            color.g = float(g)
            color.b = float(b)
            color.a = 1.0  # Alpha (transparency) value
            
            # Add to the pattern message
            pattern_msg.rgb_vals.append(color)
        
        self.led_pattern_pub.publish(pattern_msg)
    
    def move_forward(self, speed=0.2):
        """Move forward at specified speed"""
        cmd = Twist2DStamped()
        cmd.v = speed
        cmd.omega = 0.0
        self.cmd_vel_pub.publish(cmd)
    
    def turn_right(self):
        """Execute a right turn (90 degrees)"""
        # Set angular velocity for right turn
        cmd = Twist2DStamped()
        cmd.v = self.turn_speed
        cmd.omega = -3.0  # Negative for right turn
        self.cmd_vel_pub.publish(cmd)
    
    def turn_left(self):
        """Execute a left turn (90 degrees)"""
        # Set angular velocity for left turn
        cmd = Twist2DStamped()
        cmd.v = self.turn_speed
        cmd.omega = 3.0  # Positive for left turn
        self.cmd_vel_pub.publish(cmd)
    
    def stop(self):
        """Stop the robot"""
        cmd = Twist2DStamped()
        cmd.v = 0.0
        cmd.omega = 0.0
        self.cmd_vel_pub.publish(cmd)
    
    def state_machine_loop(self, event):
        """Main state machine loop for robot behavior"""
        current_time = rospy.get_time()
        state_duration = current_time - self.state_start_time
        
        # Publish current state
        self.state_pub.publish(self.state)
        
        # State machine logic
        if self.state == self.STATE_SEARCHING:
            # Move forward while searching for lines
            self.move_forward(self.normal_speed)
            
            # If we detect a line at sufficient distance, start approaching
            if self.detected_color and self.detected_distance >= self.detection_distance:
                self.log(f"Detected {self.detected_color} line at distance {self.detected_distance:.1f} cm")
                
                # Transition to approaching state
                self.state = self.STATE_APPROACHING
                self.state_start_time = current_time
        
        elif self.state == self.STATE_APPROACHING:
            # Move more slowly toward the line
            self.move_forward(self.approach_speed)
            
            # If we're close enough, stop
            if self.detected_distance <= self.stopping_distance:
                self.log(f"Stopping at {self.detected_color} line")
                
                # Transition to stopping state
                self.state = self.STATE_STOPPING
                self.state_start_time = current_time
                self.stop()
                self.set_led_pattern("brake")
        
        elif self.state == self.STATE_STOPPING:
            # Stop for 3-5 seconds
            if state_duration >= self.stop_duration:
                color = self.detected_color
                
                if color == "blue":
                    # Blue line: Right turn with right signal
                    self.log("Starting right turn (blue line)")
                    self.set_led_pattern("right")
                    self.state = self.STATE_TURNING_RIGHT
                    self.state_start_time = current_time
                    self.turn_right()
                    
                elif color == "red":
                    # Red line: Move forward
                    self.log("Moving straight (red line)")
                    self.set_led_pattern("brake")
                    self.state = self.STATE_MOVING_STRAIGHT
                    self.state_start_time = current_time
                    self.move_forward(self.normal_speed)
                    
                elif color == "green":
                    # Green line: Left turn with left signal
                    self.log("Starting left turn (green line)")
                    self.set_led_pattern("left")
                    self.state = self.STATE_TURNING_LEFT
                    self.state_start_time = current_time
                    self.turn_left()
        
        elif self.state == self.STATE_TURNING_RIGHT:
            # Executing right turn (blue line)
            if state_duration >= self.right_turn_duration:
                self.log("Right turn completed, continuing forward")
                self.state = self.STATE_CONTINUE
                self.state_start_time = current_time
                self.set_led_pattern("off")
                self.move_forward(self.normal_speed)
        
        elif self.state == self.STATE_TURNING_LEFT:
            # Executing left turn (green line)
            if state_duration >= self.left_turn_duration:
                self.log("Left turn completed, continuing forward")
                self.state = self.STATE_CONTINUE
                self.state_start_time = current_time
                self.set_led_pattern("off")
                self.move_forward(self.normal_speed)
        
        elif self.state == self.STATE_MOVING_STRAIGHT:
            # Moving straight after red line (at least 30 cm)
            time_to_travel = self.straight_distance / self.normal_speed
            if state_duration >= time_to_travel:
                self.log("Straight movement completed, continuing forward")
                self.state = self.STATE_CONTINUE
                self.state_start_time = current_time
                self.set_led_pattern("off")
        
        elif self.state == self.STATE_CONTINUE:
            # Continue moving forward
            # Reset to searching state after 5 seconds to detect new lines
            if state_duration >= 5.0:
                self.state = self.STATE_SEARCHING
                self.state_start_time = current_time
                self.detected_color = None
                self.detected_distance = 0
    
    def stop_robot(self):
        """Stop the robot completely"""
        self.stop()
        self.set_led_pattern("off")

if __name__ == '__main__':
    # Initialize the node
    vision_node = ComputerVisionNode(node_name='computer_vision_node')
    
    try:
        # Spin until interrupted
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to stop the robot when the node is shut down
        vision_node.stop_robot()