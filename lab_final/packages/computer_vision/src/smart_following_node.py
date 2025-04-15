#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String
from enum import Enum
import time


class OperationMode(Enum):
    LANE_FOLLOWING = 0
    LEADER_FOLLOWING = 1


class SmartFollowingNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(SmartFollowingNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Get vehicle name
        self.vehicle_name = os.environ['VEHICLE_NAME']

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Operation mode settings
        self.operation_mode = OperationMode.LANE_FOLLOWING

        # Leader detection parameters
        self.leader_detected = False
        self.leader_x_position = 0
        self.leader_width = 0
        self.frames_since_last_detection = 0
        self.max_lost_frames = 10

        # HSV thresholds for blue Duckiebot detection
        self.blue_lower = np.array([100, 70, 50])
        self.blue_upper = np.array([130, 255, 255])

        # Area constraints for leader detection
        self.min_bot_area = 500
        self.max_bot_area = 100000

        # Leader following parameters
        self.target_distance = 0.7  # Target distance in meters
        self.stop_distance = 0.25  # Minimum safe distance

        # PID parameters for distance control
        self.kp_distance = 0.5
        self.ki_distance = 0.0
        self.kd_distance = 0.1
        self.distance_error_integral = 0
        self.prev_distance_error = 0

        # Distance estimation parameters
        self.real_duckiebot_width = 0.13  # meters
        self.camera_focal_length = 220  # pixels

        # Lane detection parameters
        # HSV thresholds for yellow lane
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])

        # HSV thresholds for white lane
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 60, 255])

        # Lane controller parameters
        self.lane_kp = 0.035
        self.lane_kd = 0.01
        self.lane_ki = 0.001

        # Lane controller state
        self.lane_error = 0
        self.lane_last_error = 0
        self.lane_integral = 0
        self.lane_derivative = 0
        self.lane_last_time = rospy.get_time()

        # Speed parameters
        self.normal_speed = 0.4
        self.turn_speed = 0.3
        self.current_speed = self.normal_speed

        # Region of interest for lane detection
        self.roi_top = 250
        self.roi_height = 120
        self.roi_width = 640

        # Image parameters
        self.image_center_x = 320  # Will be updated with actual image width

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
        self.vis_pub = rospy.Publisher(
            f"/{self.vehicle_name}/smart_following/image/compressed",
            CompressedImage,
            queue_size=1
        )

        self.cmd_vel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        self.status_pub = rospy.Publisher(
            f"/{self.vehicle_name}/smart_following/status",
            String,
            queue_size=1
        )

        # Timer for control loop
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        self.log("Smart following node initialized")

    def camera_callback(self, msg):
        """Process the camera image for lane and leader detection"""
        try:
            # Convert compressed image to CV image
            img = self.bridge.compressed_imgmsg_to_cv2(msg)

            # Update image center based on actual image size
            h, w = img.shape[:2]
            self.image_center_x = w // 2

            # Make a copy for visualization
            vis_img = img.copy()

            # Detect leader duckiebot (in the upper part of the image)
            self.detect_leader_duckiebot(img, vis_img)

            # Extract ROI for lane detection (in the lower part of the image)
            roi = img[self.roi_top:self.roi_top + self.roi_height, 0:self.roi_width]
            roi_vis = vis_img[self.roi_top:self.roi_top + self.roi_height, 0:self.roi_width]

            # Detect lanes
            yellow_mask, yellow_center, yellow_contour = self.detect_lane(roi, 'yellow')
            white_mask, white_center, white_contour = self.detect_lane(roi, 'white')

            # Visualize lane detection
            if yellow_contour is not None:
                cv2.drawContours(roi_vis, [yellow_contour], -1, (0, 255, 255), 2)
                if yellow_center is not None:
                    cv2.circle(roi_vis, (yellow_center, roi.shape[0] // 2), 5, (0, 255, 255), -1)

            if white_contour is not None:
                cv2.drawContours(roi_vis, [white_contour], -1, (255, 255, 255), 2)
                if white_center is not None:
                    cv2.circle(roi_vis, (white_center, roi.shape[0] // 2), 5, (255, 255, 255), -1)

            # Calculate lane error
            self.calculate_lane_error(yellow_center, white_center, roi.shape[1])

            # Draw lane error visualization
            self.draw_lane_error_visualization(roi_vis)

            # Update operation mode based on leader detection
            self.update_operation_mode()

            # Add data overlay to visualization
            self.add_data_overlay(vis_img)

            # Publish visualization
            if self.vis_pub.get_num_connections() > 0:
                vis_msg = self.bridge.cv2_to_compressed_imgmsg(vis_img)
                self.vis_pub.publish(vis_msg)

        except Exception as e:
            self.logerr(f"Error processing camera image: {str(e)}")

    def detect_leader_duckiebot(self, img, vis_img):
        """Detect the leader duckiebot by looking for blue objects"""
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create a mask to isolate blue regions
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=1)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=3)

        # Add mask to visualization in smaller window
        h, w = blue_mask.shape
        scaled_mask = cv2.resize(blue_mask, (w // 4, h // 4))
        scaled_mask_color = cv2.cvtColor(scaled_mask, cv2.COLOR_GRAY2BGR)

        if h // 4 < vis_img.shape[0] and w // 4 < vis_img.shape[1]:
            vis_img[0:h // 4, 0:w // 4] = scaled_mask_color

        # Track detection status
        prev_detected = self.leader_detected
        self.leader_detected = False

        # Find contours in the blue mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter by size
            valid_contours = [c for c in contours if self.min_bot_area < cv2.contourArea(c) < self.max_bot_area]

            # Sort contours by area (largest first)
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

            # Draw all valid contours for debugging
            for c in valid_contours:
                cv2.drawContours(vis_img, [c], -1, (0, 255, 0), 1)

            # Examine the contours
            for i, contour in enumerate(valid_contours[:5]):
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate aspect ratio
                aspect_ratio = float(w) / h

                # Check if it has reasonable dimensions for a Duckiebot
                if w > 40 and h > 30 and 1.0 < aspect_ratio < 3.0:
                    # Calculate the percentage of the contour area that actually contains blue
                    bot_region = hsv[y:y + h, x:x + w]
                    blue_pixels = cv2.inRange(bot_region, self.blue_lower, self.blue_upper)
                    blue_percentage = np.sum(blue_pixels > 0) / (w * h)

                    # Only accept if a significant portion is blue
                    if blue_percentage > 0.3:
                        # This is likely our leader Duckiebot
                        self.leader_detected = True
                        self.leader_x_position = x + w // 2
                        self.leader_width = w

                        # Reset frames counter
                        self.frames_since_last_detection = 0

                        # Draw the leader indicator
                        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
                        cv2.putText(vis_img, "LEADER",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # Estimate distance using camera model
                        distance = self.estimate_distance(w)
                        cv2.putText(vis_img, f"Est. Distance: {distance:.2f}m",
                                    (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        break  # Stop after finding a valid leader

        # If no leader detected in this frame, increment counter
        if not self.leader_detected:
            self.frames_since_last_detection += 1

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

    def calculate_lane_error(self, yellow_center, white_center, img_width):
        """Calculate the error for lane following based on detected lanes"""
        # Center of the image
        img_center = img_width // 2

        # Check if we have both lane markings
        if yellow_center is not None and white_center is not None:
            # Both lanes detected - aim for the center
            lane_center = (yellow_center + white_center) // 2
            self.lane_error = lane_center - img_center

            # Adjust speed based on lane width
            lane_width = abs(white_center - yellow_center)
            if lane_width < 150:  # Lane is narrower than expected
                self.current_speed = self.turn_speed  # Slow down for turns
            else:
                self.current_speed = self.normal_speed  # Regular speed

        elif yellow_center is not None:
            # Only yellow lane detected - stay a fixed distance to the right
            self.lane_error = yellow_center - (img_center - 100)  # Offset by approx lane width/2
            self.current_speed = self.turn_speed  # Slow down when only one lane is visible

        elif white_center is not None:
            # Only white lane detected - stay a fixed distance to the left
            self.lane_error = white_center - (img_center + 100)  # Offset by approx lane width/2
            self.current_speed = self.turn_speed  # Slow down when only one lane is visible

    def draw_lane_error_visualization(self, img):
        """Add visual indicators for lane error and target path"""
        h, w = img.shape[:2]

        # Draw center of image (reference)
        center_x = w // 2
        cv2.line(img, (center_x, 0), (center_x, h), (0, 0, 255), 1)

        # Draw error line
        if self.lane_error is not None:
            # Calculate position based on error
            error_pos = center_x + self.lane_error
            if 0 <= error_pos < w:
                cv2.line(img, (error_pos, 0), (error_pos, h), (255, 0, 0), 2)

                # Draw a line connecting center to error position
                cv2.line(img, (center_x, h // 2), (error_pos, h // 2), (0, 255, 0), 2)

    def estimate_distance(self, apparent_width):
        """Estimate distance to leader using camera model"""
        # Use simple pinhole camera model: distance = (real_width * focal_length) / apparent_width
        if apparent_width > 0:
            distance = (self.real_duckiebot_width * self.camera_focal_length) / apparent_width
            return distance
        return 999.9  # Large default value if width is zero

    def update_operation_mode(self):
        """Update the operation mode based on leader detection"""
        previous_mode = self.operation_mode

        if self.leader_detected:
            self.operation_mode = OperationMode.LEADER_FOLLOWING
        elif self.frames_since_last_detection > self.max_lost_frames:
            self.operation_mode = OperationMode.LANE_FOLLOWING

        # Log mode changes
        if previous_mode != self.operation_mode:
            self.log(f"Mode changed: {previous_mode.name} -> {self.operation_mode.name}")

    def control_loop(self, event=None):
        """Main control loop for combined lane and leader following"""
        # Calculate lane-following commands
        lane_v, lane_omega = self.calculate_lane_following_commands()

        # If leader is detected, calculate leader-following parameters
        if self.operation_mode == OperationMode.LEADER_FOLLOWING:
            # Calculate adaptive velocity based on leader distance
            distance = self.estimate_distance(self.leader_width)
            leader_v = self.calculate_adaptive_velocity(distance)

            # Use lane following for steering, but adapt speed for leader
            v = leader_v
            omega = lane_omega

            # Publish status
            status_msg = (f"Leader following - Distance: {distance:.2f}m, "
                          f"v: {v:.2f}, omega: {omega:.2f}")
        else:
            # Pure lane following
            v = lane_v
            omega = lane_omega

            # Publish status
            status_msg = (f"Lane following - Error: {self.lane_error:.1f}, "
                          f"v: {v:.2f}, omega: {omega:.2f}")

        # Publish control command
        self.publish_command(v, omega)

        # Publish status
        self.status_pub.publish(status_msg)

    def calculate_lane_following_commands(self):
        """Calculate the control commands for lane following"""
        # Calculate dt
        current_time = rospy.get_time()
        dt = current_time - self.lane_last_time
        if dt == 0:
            return 0.0, 0.0
        self.lane_last_time = current_time

        # Calculate derivative
        self.lane_derivative = (self.lane_error - self.lane_last_error) / dt

        # Calculate integral with anti-windup
        self.lane_integral += self.lane_error * dt

        # Anti-windup: Limit integral term
        if self.lane_integral > 100:
            self.lane_integral = 100
        elif self.lane_integral < -100:
            self.lane_integral = -100

        # PID controller for omega
        omega = -self.lane_kp * self.lane_error - self.lane_kd * self.lane_derivative - self.lane_ki * self.lane_integral

        # Limit the maximum omega to prevent excessive rotation
        max_omega = 8.0
        omega = max(min(omega, max_omega), -max_omega)

        # Set velocity
        v = self.current_speed

        # Save error for next iteration
        self.lane_last_error = self.lane_error

        return v, omega

    def calculate_adaptive_velocity(self, distance):
        """Calculate velocity based on distance to leader"""
        if distance < self.stop_distance:
            # Too close - stop or back up slightly
            return -0.1
        elif distance < self.target_distance * 0.8:
            # Closer than target - slow down
            return 0.1
        elif distance < self.target_distance:
            # Approaching target - moderate speed
            return 0.2
        elif distance < self.target_distance * 1.2:
            # Near target - normal speed
            return 0.3
        else:
            # Further than target - faster to catch up (but not too fast)
            return min(0.4, self.current_speed)

    def publish_command(self, v, omega):
        """Publish control command to Duckiebot"""
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.cmd_vel_pub.publish(msg)

    def add_data_overlay(self, vis_img):
        """Add information overlay to the visualization image"""
        # Display operation mode
        mode_text = f"MODE: {self.operation_mode.name}"
        if self.operation_mode == OperationMode.LANE_FOLLOWING:
            mode_color = (0, 255, 0)  # Green
        else:  # LEADER_FOLLOWING
            mode_color = (0, 165, 255)  # Orange

        cv2.putText(vis_img, mode_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Display leader detection status
        if self.leader_detected:
            status_text = "LEADER DETECTED"
            distance = self.estimate_distance(self.leader_width)
            distance_text = f"Distance: {distance:.2f}m"

            cv2.putText(vis_img, status_text,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_img, distance_text,
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            status_text = "NO LEADER DETECTED"
            cv2.putText(vis_img, status_text,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display lane following information
        lane_text = f"Lane Error: {self.lane_error:.1f}"
        cv2.putText(vis_img, lane_text,
                    (10, vis_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def stop(self):
        """Stop the robot by publishing zero velocity"""
        self.publish_command(0, 0)
        self.log("Smart following node stopped")


if __name__ == '__main__':
    # Initialize the node
    smart_follower = SmartFollowingNode(node_name='smart_following_node')

    try:
        # Spin until interrupted
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to stop the robot when the node is shut down
        smart_follower.stop()