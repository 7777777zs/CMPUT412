#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String, ColorRGBA
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern, Twist2DStamped


class DuckDetectionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(DuckDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Get vehicle name from environment variable
        self.vehicle_name = os.environ['VEHICLE_NAME']

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Duck detection parameters
        self.duck_detection_threshold = 50  # Minimum area to consider a detection valid
        self.debug_mode = True  # Enable debug output

        # HSV thresholds specifically for orange ducks
        self.duck_lower = np.array([5, 100, 100])
        self.duck_upper = np.array([25, 255, 255])

        # ROI parameters - expanded to see more of the scene
        self.roi_top = 100  # Higher up to catch ducks earlier
        self.roi_height = 300  # Larger height to capture more of the scene
        self.roi_width = 640  # Full width

        # Define color presets
        self.led_color_dict = {
            "red": [1.0, 0.0, 0.0, 1.0],    # Red with full opacity
            "green": [0.0, 1.0, 0.0, 1.0],  # Green with full opacity
            "blue": [0.0, 0.0, 1.0, 1.0],   # Blue with full opacity
            "white": [1.0, 1.0, 1.0, 1.0],  # White with full opacity
            "yellow": [1.0, 1.0, 0.0, 1.0],  # Yellow with full opacity
            "off": [0.0, 0.0, 0.0, 1.0],    # Off with full opacity
        }

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

        # 5. Command publisher to stop the robot
        self.cmd_vel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        # Add duck detection threshold debug output
        self.log(
            f"Duck detection threshold: {self.duck_detection_threshold} pixels")
        self.log(
            f"Duck HSV range: Lower {self.duck_lower}, Upper {self.duck_upper}")
        self.log(
            f"Publishing visualization to: /{self.vehicle_name}/duck_detection_node/image/compressed")

    def camera_callback(self, msg):
        """Process the camera image to detect ducks"""
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

            # Debug info
            if self.debug_mode:
                all_contours, _ = cv2.findContours(
                    duck_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.log(
                    f"Duck detected: {duck_detected}, filtered contours: {len(duck_contours)}, all contours: {len(all_contours)}")
                if duck_contours:
                    areas_str = ", ".join(
                        [f"{area:.1f}" for area in duck_areas])
                    self.log(f"Duck areas: {areas_str}")
                elif all_contours:
                    all_areas = [cv2.contourArea(c) for c in all_contours]
                    all_areas_str = ", ".join(
                        [f"{area:.1f}" for area in all_areas])
                    self.log(
                        f"All contour areas (before filtering): {all_areas_str}")

            # Set LED pattern based on duck detection status
            if duck_detected:
                # Red LEDs for duck detection
                self.set_led_pattern("red")
                self.stop_robot()  # Stop the robot when ducks are detected

                # Publish detailed information about the ducks
                details = f"Detected {len(duck_contours)} ducks. "
                for i, (centroid, area) in enumerate(zip(duck_centroids, duck_areas)):
                    details += f"Duck {i+1}: position ({centroid[0]}, {centroid[1]}), size {area:.1f} px²; "

                self.details_pub.publish(details)
            else:
                # White LEDs when no ducks are detected
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

            # Publish duck detection status
            self.detection_pub.publish(duck_detected)

            # Create debug visualization
            debug_img = self.create_debug_visualization(vis_img, duck_mask)

            # Publish debug visualization
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            debug_msg.header = msg.header
            self.vis_pub.publish(debug_msg)

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

    def stop_robot(self):
        """Stop the robot by publishing zero velocity"""
        cmd = Twist2DStamped()
        cmd.v = 0
        cmd.omega = 0
        self.cmd_vel_pub.publish(cmd)

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


if __name__ == '__main__':
    # Initialize the node
    duck_detection_node = DuckDetectionNode(node_name='duck_detection_node')

    try:
        # Spin until interrupted
        rospy.spin()
        # FIX: Use lambda to properly call the shutdown handler with default arg
        rospy.on_shutdown(lambda: duck_detection_node.set_led_pattern())
    except KeyboardInterrupt:
        pass
    finally:
        # Turn off LEDs when shutting down - now works without parameter
        duck_detection_node.set_led_pattern()
