#!/usr/bin/env python3

# import required libraries
from os import uname, environ
from re import match
from math import pi
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped, LEDPattern, Twist2DStamped
from std_msgs.msg import Float32, Int16, Bool, String, ColorRGBA
from enum import Enum


class SequenceState(Enum):
    LOOKING_FOR_BLUE_LINES = 1
    APPROACHING_BLUE_LINE = 2
    WAITING_FOR_RED_LINE = 3
    STOPPED_AT_RED_LINE = 4
    WAITING_AFTER_SEQUENCE = 5
    DUCK_DETECTED = 6      # New state for when a duck is detected


class DuckDetectionNode(DTROS):
    def __init__(self, node_name):
        super(DuckDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)

        # Initialize parameters
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname

        # Get vehicle name from environment variable (for duck detection)
        # This should match self._vehicle_name, but keeping both for compatibility
        self.vehicle_name = environ.get('VEHICLE_NAME', self._vehicle_name)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Duck detection parameters
        # Minimum area to consider a detection valid (increased from 50)
        self.duck_detection_threshold = 2000
        self.debug_mode = True  # Enable debug output

        # HSV thresholds specifically for orange ducks
        self.duck_lower = np.array([5, 100, 100])
        self.duck_upper = np.array([25, 255, 255])

        # ROI parameters - expanded to see more of the scene
        self.roi_top = 100  # Higher up to catch ducks earlier
        self.roi_height = 300  # Larger height to capture more of the scene
        self.roi_width = 640  # Full width

        # Define color presets for LEDs
        self.led_color_dict = {
            "red": [1.0, 0.0, 0.0, 1.0],    # Red with full opacity
            "green": [0.0, 1.0, 0.0, 1.0],  # Green with full opacity
            "blue": [0.0, 0.0, 1.0, 1.0],   # Blue with full opacity
            "white": [1.0, 1.0, 1.0, 1.0],  # White with full opacity
            "yellow": [1.0, 1.0, 0.0, 1.0],  # Yellow with full opacity
            "off": [0.0, 0.0, 0.0, 1.0],    # Off with full opacity
        }

        # Encoder data and flags
        self._ticks_left = None
        self._ticks_right = None
        self._start_left = 0
        self._start_right = 0
        self._just_started_left = True
        self._just_started_right = True
        self._velocity = 0.20

        # Line detection variables
        self._red_line_distance = float('inf')
        self._blue_line_double_distance = float(
            'inf')  # Distance to double blue lines
        # Distance to nearest blue line
        self._blue_line_distance = float('inf')

        # Duck detection status
        self._duck_detected = False
        self._duck_timer = None
        self._duck_stop_duration = 5.0  # Stop for 5 seconds when duck detected

        # State tracking variables
        self._current_state = SequenceState.LOOKING_FOR_BLUE_LINES
        self._previous_state = None  # To track state transitions
        self._sequence_start_time = None
        self._stop_start_time = None
        # Wait 1 second after red line disappears
        self._wait_after_sequence_duration = 1.0

        # Debug flag for enhanced logging
        self._debug = True

        # Log initial state
        rospy.loginfo(f"INITIAL STATE: {self._current_state.name}")

        # Subscribe to encoder topics
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self.sub_left = rospy.Subscriber(
            self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(
            self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # Subscribe to line detection topics
        self.red_distance_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/red_line_distance", Float32, self.red_distance_callback)

        # Subscribe to blue line detection topics
        self.blue_distance_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/blue_line_distance", Float32, self.blue_distance_callback)

        self.blue_double_distance_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/blue_line_double_distance", Float32, self.blue_double_distance_callback)

        # Subscribe to camera image for duck detection
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.camera_sub = rospy.Subscriber(
            self.camera_topic,
            CompressedImage,
            self.camera_callback,
            queue_size=1,
            buff_size=10000000
        )

        # Publishers
        # 1. Visualization publisher (with duck detection boxes)
        self.vis_pub = rospy.Publisher(
            f"/{self.vehicle_name}/duck_detection_node/image/compressed",
            CompressedImage,
            queue_size=1
        )

        # 2. Duck detection status publisher (Boolean)
        self.detection_pub = rospy.Publisher(
            f"/{self.vehicle_name}/duck_detection_node/detected",
            Bool,
            queue_size=1
        )

        # 3. Duck detection details publisher
        self.details_pub = rospy.Publisher(
            f"/{self.vehicle_name}/duck_detection_node/details",
            String,
            queue_size=1
        )

        # 4. LED pattern publisher
        self.led_pattern_pub = rospy.Publisher(
            f"/{self.vehicle_name}/led_emitter_node/led_pattern",
            LEDPattern,
            queue_size=1
        )

        # 5. Command publisher to stop the robot - now using car_cmd_switch_node
        self.cmd_vel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        # Get parameters
        self._radius = rospy.get_param(
            f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)  # Default radius
        self._resolution_left = rospy.get_param(
            f'/{self._vehicle_name}/left_wheel_encoder_node/resolution', 135)
        self._resolution_right = rospy.get_param(
            f'/{self._vehicle_name}/right_wheel_encoder_node/resolution', 135)

        # this is just equal to 2dw, so no need to change it for now
        self._baseline = rospy.get_param(
            f'/{self._vehicle_name}/kinematics_node/baseline', 0.1)

        if self._resolution_right != self._resolution_left:
            rospy.logwarn(
                "The resolutions of the left and right wheels do not match!")

        self._dist_per_tick = self._radius * 2 * pi / self._resolution_left

        # Initialize velocities
        self._vel_left = self._velocity
        self._vel_right = self._velocity

        # Create a timer for periodic checks
        self._timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        # Add duck detection threshold debug output
        self.log(
            f"Duck detection threshold: {self.duck_detection_threshold} pixels")
        self.log(
            f"Duck HSV range: Lower {self.duck_lower}, Upper {self.duck_upper}")
        self.log(
            f"Publishing visualization to: /{self.vehicle_name}/duck_detection_node/image/compressed")

    def change_state(self, new_state):
        """Helper method to change state with logging"""
        if self._current_state != new_state:
            self._previous_state = self._current_state
            self._current_state = new_state
            rospy.loginfo(
                f"STATE TRANSITION: {self._previous_state.name} -> {self._current_state.name}")
            return True
        return False

    def callback_left(self, data):
        """Left wheel encoder callback"""
        rospy.loginfo_once(f"Left encoder resolution: {data.resolution}")
        self._ticks_left = data.data
        if self._just_started_left:
            self._start_left = self._ticks_left
            self._just_started_left = False

    def callback_right(self, data):
        """Right wheel encoder callback"""
        rospy.loginfo_once(f"Right encoder resolution: {data.resolution}")
        self._ticks_right = data.data
        if self._just_started_right:
            self._start_right = self._ticks_right
            self._just_started_right = False

    def red_distance_callback(self, msg):
        """Callback for red line distance"""
        self._red_line_distance = msg.data
        if self._debug and self._red_line_distance < 0.2:
            rospy.loginfo_throttle(
                1.0, f"Red line distance: {self._red_line_distance:.3f}m")

    def blue_distance_callback(self, msg):
        """Callback for single blue line distance"""
        self._blue_line_distance = msg.data
        if self._debug and self._blue_line_distance < 0.2:
            rospy.loginfo_throttle(
                1.0, f"Blue line distance: {self._blue_line_distance:.3f}m")

    def blue_double_distance_callback(self, msg):
        """Callback for double blue lines distance"""
        self._blue_line_double_distance = msg.data
        if self._debug and self._blue_line_double_distance < float('inf'):
            rospy.loginfo_throttle(
                1.0, f"Double blue line distance: {self._blue_line_double_distance:.3f}m")

    def camera_callback(self, msg):
        """Process the camera image to detect ducks and visualize blue lines"""
        try:
            # Convert compressed image to CV image
            img = self.bridge.compressed_imgmsg_to_cv2(msg)

            # Crop to region of interest
            roi = img[self.roi_top:self.roi_top +
                      self.roi_height, 0:self.roi_width]

            # Make a copy for visualization
            vis_img = roi.copy()

            # Detect ducks
            duck_mask = self.create_duck_mask(roi)
            duck_detected, duck_contours, duck_centroids, duck_areas = self.detect_duck_contours(
                duck_mask.copy())

            # Update duck detection status
            previous_duck_state = self._duck_detected
            self._duck_detected = duck_detected

            # Debug info
            if self.debug_mode and duck_detected:
                all_contours, _ = cv2.findContours(
                    duck_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.log(
                    f"Duck detected: {duck_detected}, filtered contours: {len(duck_contours)}, all contours: {len(all_contours)}")
                if duck_contours:
                    areas_str = ", ".join(
                        [f"{area:.1f}" for area in duck_areas])
                    self.log(f"Duck areas: {areas_str}")

            # If duck detected state changed, handle transition
            if previous_duck_state != self._duck_detected:
                if self._duck_detected:
                    rospy.loginfo(
                        "Duck detected! Transitioning to duck detected state")
                    self._previous_state_before_duck = self._current_state
                    self.change_state(SequenceState.DUCK_DETECTED)
                    self._stop_start_time = rospy.Time.now()
                    # Set red LEDs when duck detected
                    self.set_led_pattern("red")
                else:
                    rospy.loginfo("Duck no longer detected")
                    # Set white LEDs when duck not detected
                    self.set_led_pattern("white")

            # Draw detection boxes on visualization image
            if duck_contours:
                # Draw all contours
                cv2.drawContours(vis_img, duck_contours, -1, (0, 0, 255), 2)

                # Draw bounding boxes and labels
                for i, contour in enumerate(duck_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(vis_img, (x, y),
                                  (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(vis_img, f"Duck {i+1}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Draw centroids safely
                    if i < len(duck_centroids):
                        centroid = duck_centroids[i]
                        if isinstance(centroid, tuple) and len(centroid) == 2:
                            cx, cy = centroid
                            cv2.circle(vis_img, (int(cx), int(cy)),
                                       5, (0, 255, 0), -1)

            # Add text indicating whether ducks are detected
            detection_text = "DUCKS DETECTED!" if duck_detected else "No ducks detected"
            cv2.putText(vis_img, detection_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if duck_detected else (0, 255, 0), 2)

            # Visualize blue line information
            blue_line_text = f"Blue line: {self._blue_line_distance:.2f}m"
            blue_double_text = f"Double blue: {self._blue_line_double_distance:.2f}m"
            red_line_text = f"Red line: {self._red_line_distance:.2f}m"

            # Position the text lower in the image to avoid overlap
            cv2.putText(vis_img, blue_line_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(vis_img, blue_double_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(vis_img, red_line_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Add current state to visualization
            state_text = f"State: {self._current_state.name}"
            cv2.putText(vis_img, state_text, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Publish duck detection status
            self.detection_pub.publish(duck_detected)

            # Create debug visualization
            debug_img = self.create_debug_visualization(vis_img, duck_mask)

            # Publish debug visualization if there are subscribers
            if self.vis_pub.get_num_connections() > 0:
                debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                debug_msg.header = msg.header
                self.vis_pub.publish(debug_msg)

            # Publish details about ducks
            if duck_detected and duck_contours:
                details = f"Detected {len(duck_contours)} ducks. "
                for i, (centroid, area) in enumerate(zip(duck_centroids, duck_areas)):
                    details += f"Duck {i+1}: position ({centroid[0]}, {centroid[1]}), size {area:.1f} px²; "
                self.details_pub.publish(details)

        except Exception as e:
            self.logerr(f"Error processing camera image: {str(e)}")

    def create_duck_mask(self, img):
        """Create a binary mask for duck detection"""
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply color mask for ducks
        duck_mask = cv2.inRange(hsv, self.duck_lower, self.duck_upper)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        duck_mask = cv2.erode(duck_mask, kernel, iterations=1)
        # More dilation to connect nearby parts
        duck_mask = cv2.dilate(duck_mask, kernel, iterations=2)

        return duck_mask

    def detect_duck_contours(self, mask):
        """Detect duck contours in the binary mask"""
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size to eliminate small noise
        valid_contours = [c for c in contours if cv2.contourArea(
            c) > self.duck_detection_threshold]

        # Calculate centroids and areas for valid contours
        centroids = []
        areas = []
        for contour in valid_contours:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

                # Calculate area
                area = cv2.contourArea(contour)
                areas.append(area)
            else:
                # If we can't calculate moments, use bounding box center
                x, y, w, h = cv2.boundingRect(contour)
                centroids.append((x + w//2, y + h//2))
                areas.append(w * h)

        # Determine if ducks are detected
        ducks_detected = len(valid_contours) > 0

        return ducks_detected, valid_contours, centroids, areas

    def create_debug_visualization(self, vis_img, duck_mask):
        """Create a debug visualization with the original image and mask"""
        # Create a 3-channel version of the mask for visualization
        # FIX: Ensure duck_mask is properly binary before conversion
        duck_mask_safe = duck_mask.copy()
        _, duck_mask_safe = cv2.threshold(
            duck_mask_safe, 127, 255, cv2.THRESH_BINARY)
        duck_mask_color = cv2.cvtColor(duck_mask_safe, cv2.COLOR_GRAY2BGR)

        # Resize to match the visualization image if needed
        if duck_mask_color.shape != vis_img.shape:
            duck_mask_color = cv2.resize(
                duck_mask_color, (vis_img.shape[1], vis_img.shape[0]))

        # Create a split view with original and mask
        h, w = vis_img.shape[:2]
        debug_img = np.zeros((h, w*2, 3), dtype=np.uint8)
        debug_img[:, :w] = vis_img
        debug_img[:, w:] = duck_mask_color

        # Add label to mask view
        cv2.putText(debug_img, "Duck Mask", (w+10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw all contours on the mask view
        # FIX: Use the safe mask for finding contours
        all_contours, _ = cv2.findContours(
            duck_mask_safe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if all_contours:
            cv2.drawContours(debug_img[:, w:],
                             all_contours, -1, (0, 0, 255), 2)

        return debug_img

    def set_led_pattern(self, color_pattern="off"):
        """Set LED pattern using predefined colors or a list of colors"""
        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()

        # Create a list of 5 ColorRGBA messages (one for each LED)
        rgb_vals = []

        # Handle string color patterns
        if isinstance(color_pattern, str):
            if color_pattern == "off":
                # All LEDs off
                for _ in range(5):
                    color_rgba = ColorRGBA()
                    color_rgba.r = 0.0
                    color_rgba.g = 0.0
                    color_rgba.b = 0.0
                    color_rgba.a = 1.0
                    rgb_vals.append(color_rgba)
            elif color_pattern == "left":
                # Left turn signal (leftmost LEDs in yellow)
                for i in range(5):
                    color_rgba = ColorRGBA()
                    if i < 2:  # First two LEDs are yellow
                        color_rgba.r = 1.0
                        color_rgba.g = 1.0
                        color_rgba.b = 0.0
                    else:  # Rest are off
                        color_rgba.r = 0.0
                        color_rgba.g = 0.0
                        color_rgba.b = 0.0
                    color_rgba.a = 1.0
                    rgb_vals.append(color_rgba)
            elif color_pattern == "right":
                # Right turn signal (rightmost LEDs in yellow)
                for i in range(5):
                    color_rgba = ColorRGBA()
                    if i >= 3:  # Last two LEDs are yellow
                        color_rgba.r = 1.0
                        color_rgba.g = 1.0
                        color_rgba.b = 0.0
                    else:  # Rest are off
                        color_rgba.r = 0.0
                        color_rgba.g = 0.0
                        color_rgba.b = 0.0
                    color_rgba.a = 1.0
                    rgb_vals.append(color_rgba)
            elif color_pattern == "brake" or color_pattern == "red":
                # All LEDs red
                for _ in range(5):
                    color_rgba = ColorRGBA()
                    color_rgba.r = 1.0
                    color_rgba.g = 0.0
                    color_rgba.b = 0.0
                    color_rgba.a = 1.0
                    rgb_vals.append(color_rgba)
            elif color_pattern in self.led_color_dict:
                # Use a predefined color for all LEDs
                color_values = self.led_color_dict[color_pattern]
                for _ in range(5):
                    color_rgba = ColorRGBA()
                    color_rgba.r = color_values[0]
                    color_rgba.g = color_values[1]
                    color_rgba.b = color_values[2]
                    color_rgba.a = color_values[3]
                    rgb_vals.append(color_rgba)
            else:
                # Default: all off
                for _ in range(5):
                    color_rgba = ColorRGBA()
                    color_rgba.r = 0.0
                    color_rgba.g = 0.0
                    color_rgba.b = 0.0
                    color_rgba.a = 1.0
                    rgb_vals.append(color_rgba)
        # Handle a list of RGB tuples
        elif isinstance(color_pattern, list):
            for rgb_tuple in color_pattern:
                if len(rgb_tuple) >= 3:
                    color_rgba = ColorRGBA()
                    color_rgba.r = float(rgb_tuple[0])
                    color_rgba.g = float(rgb_tuple[1])
                    color_rgba.b = float(rgb_tuple[2])
                    color_rgba.a = 1.0 if len(
                        rgb_tuple) < 4 else float(rgb_tuple[3])
                    rgb_vals.append(color_rgba)

        # Ensure we have exactly 5 LEDs (pad with off if needed)
        while len(rgb_vals) < 5:
            color_rgba = ColorRGBA()
            color_rgba.r = 0.0
            color_rgba.g = 0.0
            color_rgba.b = 0.0
            color_rgba.a = 1.0
            rgb_vals.append(color_rgba)

        # Truncate if we have more than 5
        if len(rgb_vals) > 5:
            rgb_vals = rgb_vals[:5]

        # Assign the list of ColorRGBA objects to the message
        pattern_msg.rgb_vals = rgb_vals

        # Set frequency to 0 for static colors (no flashing)
        pattern_msg.frequency = 0.0

        # Publish the pattern
        self.led_pattern_pub.publish(pattern_msg)

    def start_forward_motion(self, speed=None):
        """Start moving forward continuously using Twist2DStamped"""
        if speed is None:
            speed = self._velocity

        # Create and publish velocity command
        cmd = Twist2DStamped()
        cmd.v = speed
        cmd.omega = 0.0
        self.cmd_vel_pub.publish(cmd)

        rospy.logdebug(f"Robot moving forward at speed {speed}")

    def turn_right(self):
        """Execute a right turn (90 degrees) using Twist2DStamped"""
        # Set angular velocity for right turn
        cmd = Twist2DStamped()
        cmd.v = self._velocity * 0.75  # Reduce speed during turns
        cmd.omega = -3.0  # Negative for right turn
        self.cmd_vel_pub.publish(cmd)

        rospy.logdebug("Executing right turn")

    def turn_left(self):
        """Execute a left turn (90 degrees) using Twist2DStamped"""
        # Set angular velocity for left turn
        cmd = Twist2DStamped()
        cmd.v = self._velocity * 0.75  # Reduce speed during turns
        cmd.omega = 3.0  # Positive for left turn
        self.cmd_vel_pub.publish(cmd)

        rospy.logdebug("Executing left turn")

    def stop_robot(self):
        """Stop the robot using Twist2DStamped"""
        cmd = Twist2DStamped()
        cmd.v = 0.0
        cmd.omega = 0.0
        self.cmd_vel_pub.publish(cmd)

        rospy.loginfo("Robot stopped")

    def handle_state_machine(self):
        """Main state machine for the sequence detection with duck avoidance"""
        # Get current state
        state = self._current_state

        # State: Duck detected - takes precedence over other states
        if state == SequenceState.DUCK_DETECTED:
            # Stop the robot
            self.stop_robot()

            # Set red LEDs
            self.set_led_pattern("red")

            # Check if duck is no longer detected and wait period has elapsed
            elapsed_time = (rospy.Time.now() - self._stop_start_time).to_sec()
            if not self._duck_detected and elapsed_time >= self._duck_stop_duration:
                rospy.loginfo(
                    f"Duck no longer detected and {self._duck_stop_duration}s wait complete. Resuming operation.")
                # Return to previous state
                self.change_state(self._previous_state_before_duck)
                self.set_led_pattern("white")
            elif self._duck_detected:
                # Reset timer if duck is still detected
                self._stop_start_time = rospy.Time.now()

        # State: Looking for blue lines
        elif state == SequenceState.LOOKING_FOR_BLUE_LINES:
            # Set white LEDs during normal operation
            self.set_led_pattern("white")

            # Move forward while looking for blue lines
            self.start_forward_motion()

            # Check if we can see the double blue lines
            if self._blue_line_double_distance < float('inf'):
                rospy.loginfo(
                    f"Detected double blue lines at distance {self._blue_line_double_distance:.2f}m")
                self.change_state(SequenceState.APPROACHING_BLUE_LINE)
                # Set blue LEDs when approaching blue line
                self.set_led_pattern("blue")

        # State: Approaching blue lines
        elif state == SequenceState.APPROACHING_BLUE_LINE:
            # Continue moving forward but check if we're close to the nearest blue line
            if self._blue_line_distance < 0.1:  # Stop when close to the first blue line
                self.stop_robot()
                rospy.loginfo(
                    "Stopped at the first blue line. Now waiting for red line detection.")
                self.change_state(SequenceState.WAITING_FOR_RED_LINE)
                self._sequence_start_time = rospy.Time.now()
                # Set blue LEDs when stopped at blue line
                self.set_led_pattern("blue")

        # State: Waiting for red line
        elif state == SequenceState.WAITING_FOR_RED_LINE:
            # Check if we see a red line
            if self._red_line_distance < 0.1:
                rospy.loginfo(
                    f"Detected red line while waiting. Distance: {self._red_line_distance:.2f}m")
                self.change_state(SequenceState.STOPPED_AT_RED_LINE)
                self._stop_start_time = rospy.Time.now()
                # Set red LEDs when stopped at red line
                self.set_led_pattern("red")

            # Timeout if we've been waiting too long (optional safety feature)
            elapsed_time = (rospy.Time.now() -
                            self._sequence_start_time).to_sec()
            if elapsed_time > 10.0:  # 10 second timeout
                rospy.logwarn(
                    "Timeout waiting for red line. Restarting sequence.")
                self.change_state(SequenceState.LOOKING_FOR_BLUE_LINES)
                self.start_forward_motion()
                self.set_led_pattern("white")

        # State: Stopped at red line
        elif state == SequenceState.STOPPED_AT_RED_LINE:
            # Wait until the red line is no longer detected (maybe we've moved past it or it disappeared)
            if self._red_line_distance > 0.1:
                rospy.loginfo(
                    "Red line no longer detected. Starting wait period.")
                self.change_state(SequenceState.WAITING_AFTER_SEQUENCE)
                self._stop_start_time = rospy.Time.now()
                # Keep red LEDs on during wait period
                self.set_led_pattern("red")

        # State: Waiting after sequence
        elif state == SequenceState.WAITING_AFTER_SEQUENCE:
            # Wait for the specified duration after the red line
            elapsed_time = (rospy.Time.now() - self._stop_start_time).to_sec()
            if elapsed_time >= self._wait_after_sequence_duration:
                rospy.loginfo(
                    f"Completed {self._wait_after_sequence_duration}s wait. Restarting sequence.")
                self.change_state(SequenceState.LOOKING_FOR_BLUE_LINES)
                self.start_forward_motion()
                self.set_led_pattern("white")

    def timer_callback(self, event):
        """Timer callback for periodic checks of the state machine"""
        # Only run state machine if we're not in duck detected state or if we are
        # but the duck is no longer present
        self.handle_state_machine()

        # Periodically log the current state for easier debugging
        if self._debug:
            rospy.loginfo_throttle(
                5.0, f"CURRENT STATE: {self._current_state.name}")

            # Log the line distances and duck detection status
            rospy.loginfo_throttle(5.0,
                                   f"Distance metrics - Red: {self._red_line_distance:.2f}m, " +
                                   f"Blue: {self._blue_line_distance:.2f}m, " +
                                   f"Double Blue: {self._blue_line_double_distance:.2f}m, " +
                                   f"Duck detected: {self._duck_detected}")

    def run(self):
        """Main run function"""
        rospy.loginfo(
            "Starting Blue and Red Line Sequence Detector with Duck Detection...")

        # Wait a moment for initialization
        rospy.sleep(1)

        # Initialize LEDs to white
        self.set_led_pattern("white")

        # Start by moving forward
        self.start_forward_motion()

        # Main loop is handled by the timer callback
        rospy.spin()

    def on_shutdown(self):
        """Handle shutdown procedure"""
        # Stop the timer
        if self._timer:
            self._timer.shutdown()

        rospy.loginfo("Shutting down and stopping wheels.")
        self.stop_robot()
        self.set_led_pattern("off")
        rospy.sleep(0.5)

        # Unsubscribe from topics
        self.sub_left.unregister()
        self.sub_right.unregister()
        self.red_distance_sub.unregister()
        self.blue_distance_sub.unregister()
        self.blue_double_distance_sub.unregister()
        self.camera_sub.unregister()


if __name__ == '__main__':
    node = DuckDetectionNode(
        node_name='blue_red_sequence_with_duck_node')
    rospy.on_shutdown(node.on_shutdown)
    node.run()
