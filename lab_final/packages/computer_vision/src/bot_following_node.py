#!/usr/bin/env python3

import os
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
import time

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Range
from duckietown_msgs.msg import Twist2DStamped, BoolStamped, VehicleCorners, LEDPattern
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA, Float32, String


class CircleGridFollowerNode(DTROS):
    """
    Circle Grid Follower Node that combines detection and PID control.

    This node handles both the detection of the circle grid pattern
    and the control logic to follow the detected pattern.
    """

    def __init__(self, node_name):
        super(CircleGridFollowerNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION)

        # Get vehicle name
        self.vehicle_name = os.environ.get('VEHICLE_NAME')
        if self.vehicle_name is None:
            raise ValueError("Environment variable VEHICLE_NAME is not set")

        # Parameters for blob detection
        self.process_frequency = rospy.get_param("~process_frequency", 10.0)
        self.circlepattern_dims = rospy.get_param(
            "~circlepattern_dims", [7, 3])

        # Detection parameters
        # Time before considering pattern lost
        self.detection_timeout = rospy.Duration(0.65)
        self.last_seen = rospy.Time(0)

        # Blob detector parameters
        self.blobdetector_min_area = rospy.get_param(
            "~blobdetector_min_area", 20)
        self.blobdetector_max_area = rospy.get_param(
            "~blobdetector_max_area", 2000)
        self.blobdetector_min_dist_between_blobs = rospy.get_param(
            "~blobdetector_min_dist_between_blobs", 5)

        # Controller parameters (PID)
        # Proportional gain for distance
        self.Kp_dist = rospy.get_param("~Kp_dist", 0.013)
        # Proportional gain for angle/offset
        self.Kp_angle = rospy.get_param("~Kp_angle", -0.005)
        self.Kd = rospy.get_param("~Kd", 0.002)  # Derivative term

        # Pattern tracking variables
        self.last_pattern = (0, 0)  # (error_distance, center_offset)
        self.target_width = 100.0  # Target width for the pattern

        # Controller state variables
        self.last_error = 0
        self.last_time = rospy.get_time()

        # Initialize timing variables
        self.last_stamp = rospy.Time.now()
        self.publish_duration = rospy.Duration.from_sec(
            1.0 / self.process_frequency)

        # Detection state
        self.pattern_detected = False

        # Safety parameters
        self.stop_bot = False  # Will be set to True if obstacle detected
        self.min_velocity = 0.05
        self.max_velocity = 0.3

        # Initialize bridge
        self.bridge = CvBridge()

        # Setup blob detector
        self.setup_blob_detector()

        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Publishers
        # Detection image for visualization
        self.pub_detection_image = rospy.Publisher(
            f"/{self.vehicle_name}/circle_grid_follower/detection_image/compressed",
            CompressedImage,
            queue_size=1
        )

        # Detection status
        self.pub_detection = rospy.Publisher(
            f"/{self.vehicle_name}/circle_grid_follower/detection",
            BoolStamped,
            queue_size=1
        )

        # Command velocity
        self.pub_cmd_vel = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        # Pattern centers
        self.pub_centers = rospy.Publisher(
            f"/{self.vehicle_name}/circle_grid_follower/centers",
            VehicleCorners,
            queue_size=1
        )

        # Status publisher for debugging
        self.pub_status = rospy.Publisher(
            f"/{self.vehicle_name}/circle_grid_follower/status",
            String,
            queue_size=1
        )

        # LED pattern publisher (to turn off LEDs)
        self.led_pattern_pub = rospy.Publisher(
            f"/{self.vehicle_name}/led_emitter_node/led_pattern",
            LEDPattern,
            queue_size=1
        )

        # Safety subscribers
        # TOF sensor for obstacle detection
        self.tof_sub = rospy.Subscriber(
            f"/{self.vehicle_name}/front_center_tof_driver_node/range",
            Float32,
            self.tof_callback,
            queue_size=1
        )

        # Camera subscriber
        self.img_sub = rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size="20MB"
        )

        # Turn off LEDs to avoid reflection interference
        self.turn_off_leds()

        self.log("Circle Grid Follower Node initialized")

    def setup_blob_detector(self):
        """
        Configure blob detector for circle grid detection
        """
        params = cv2.SimpleBlobDetector_Params()

        # Thresholding parameters
        params.minThreshold = 10
        params.maxThreshold = 200
        params.thresholdStep = 10

        # Shape filter parameters
        params.filterByArea = True
        params.minArea = self.blobdetector_min_area
        params.maxArea = self.blobdetector_max_area

        params.filterByCircularity = True
        params.minCircularity = 0.7

        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Minimum distance between blobs
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs

        # Create detector
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)

    def turn_off_leds(self):
        """Turn off LEDs to avoid reflections"""
        try:
            pattern_msg = LEDPattern()
            pattern_msg.header.stamp = rospy.Time.now()

            # Create LED pattern with all LEDs off
            rgb_vals = []
            for i in range(5):
                color_rgba = ColorRGBA()
                color_rgba.r = 0.0
                color_rgba.g = 0.0
                color_rgba.b = 0.0
                color_rgba.a = 1.0
                rgb_vals.append(color_rgba)

            pattern_msg.rgb_vals = rgb_vals
            pattern_msg.frequency = 0.0

            self.led_pattern_pub.publish(pattern_msg)
            rospy.loginfo("LEDs turned off for better detection")
        except Exception as e:
            rospy.logwarn(f"Failed to turn off LEDs: {str(e)}")

    def alternative_grid_detection(self, image_cv):
        """
        Alternative detection method if the standard one fails
        This uses simpler blob detection and custom grid arrangement
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

            # Apply adaptive threshold to get binary image
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 21, 2
            )

            # Find contours in the binary image
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area and circularity
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.blobdetector_min_area or area > self.blobdetector_max_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.7:  # Same threshold as blob detector
                    continue

                filtered_contours.append(contour)

            rospy.loginfo(
                f"Alternative detection found {len(filtered_contours)} potential circles")

            # If we don't have enough contours, return None
            if len(filtered_contours) < self.circlepattern_dims[0] * self.circlepattern_dims[1]:
                return None

            # Get centers of filtered contours
            centers = []
            for contour in filtered_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))

            # TODO: Could implement grid arrangement logic here
            # For now, just provide debug visualization

            debug_img = image_cv.copy()
            for center in centers:
                cv2.circle(debug_img, center, 5, (0, 255, 0), -1)

            cv2.putText(debug_img, f"Found {len(centers)} circles", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Publish debug image
            try:
                debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                self.pub_detection_image.publish(debug_msg)
            except Exception as e:
                rospy.logerr(
                    f"Error publishing alternative debug image: {str(e)}")

            # For now, return None as we haven't implemented the grid arrangement
            return None

        except Exception as e:
            rospy.logerr(f"Error in alternative detection: {str(e)}")
            return None

    def heartbeat_callback(self, event):
        """Publish a heartbeat status message to verify the node is still running"""
        if not self.init_successful:
            rospy.logerr(
                "Node initialization was not successful. Check previous errors.")
            return

        status_msg = String()
        status_msg.data = f"Heartbeat - frames processed: {self.frame_count}, pattern detected: {self.pattern_detected}"
        self.pub_status.publish(status_msg)
        rospy.loginfo(
            f"Heartbeat - frames: {self.frame_count}, detected: {self.pattern_detected}")

    def tof_callback(self, msg):
        """Callback for TOF sensor to detect obstacles - currently disabled"""
        # We're not using the TOF sensor as requested
        pass

    def image_callback(self, msg):
        """Process incoming camera images to detect and follow circle grid"""
        now = rospy.Time.now()

        # Rate limiting
        if now - self.last_stamp < self.publish_duration:
            return

        self.last_stamp = now

        # Convert compressed image to OpenCV format
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.log(f"Error converting image: {e}", 'error')
            return

        # Detect the circle grid
        detection_result = self.detect_circle_grid(image_cv)

        # Process detection result
        if detection_result is not None:
            # Pattern detected
            error_distance, center_offset = detection_result
            self.pattern_detected = True
            self.last_seen = now

            # Publish detection status
            self.publish_detection_status(True)

            # Safety check - if obstacle detected, stop
            if self.stop_bot:
                self.stop()
                self.log("Obstacle detected, stopping robot", 'warn')
                return

            # Use PID controller to follow the pattern
            self.follow_pattern(error_distance, center_offset)

            # Publish status
            status_msg = f"Following pattern: dist_error={error_distance:.2f}, offset={center_offset:.2f}"
            self.pub_status.publish(status_msg)
        else:
            # Check if we're still within timeout
            if (now - self.last_seen) < self.detection_timeout:
                # Use last known pattern
                error_distance, center_offset = self.last_pattern
                self.pattern_detected = True

                # Publish detection status
                self.publish_detection_status(True)

                # Safety check
                if self.stop_bot:
                    self.stop()
                    return

                # Follow using last known position
                self.follow_pattern(error_distance, center_offset)

                # Publish status
                status_msg = f"Using last pattern: dist_error={error_distance:.2f}, offset={center_offset:.2f}"
                self.pub_status.publish(status_msg)
            else:
                # Pattern lost
                self.pattern_detected = False

                # Publish detection status
                self.publish_detection_status(False)

                # Stop the robot
                self.stop()

                # Publish status
                self.pub_status.publish("Pattern lost, stopping robot")

    def detect_circle_grid(self, image_cv):
        """
        Detect circle grid pattern in the image

        Args:
            image_cv: OpenCV image

        Returns:
            tuple (error_distance, center_offset) or None if not detected
        """
        # Pre-process the image
        # 1) Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # 2) Apply CLAHE for better contrast
        gray = self.clahe.apply(gray)

        # 3) Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find circle grid
        flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
        found, centers = cv2.findCirclesGrid(
            blurred,
            patternSize=tuple(self.circlepattern_dims),
            flags=flags,
            blobDetector=self.simple_blob_detector
        )

        # Create debug image
        debug_img = image_cv.copy()

        if found:
            # Calculate pattern width and offset
            xs = centers[:, 0, 0]
            pattern_width = float(np.max(xs) - np.min(xs))
            error_distance = self.target_width - pattern_width  # Positive means too far
            # Positive means pattern is to the right
            center_offset = float(np.mean(xs) - (image_cv.shape[1] / 2))

            # Draw detected pattern on debug image
            cv2.drawChessboardCorners(debug_img, tuple(
                self.circlepattern_dims), centers, found)

            # Add information text
            info_text = f"Width: {pattern_width:.1f}px, Offset: {center_offset:.1f}px"
            cv2.putText(debug_img, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Publish centers if available
            if self.pub_centers.get_num_connections() > 0:
                centers_msg = VehicleCorners()
                centers_msg.header.stamp = rospy.Time.now()
                centers_msg.detection.data = True

                for point in centers:
                    center = Point32()
                    center.x = point[0, 0]
                    center.y = point[0, 1]
                    center.z = 0
                    centers_msg.corners.append(center)

                self.pub_centers.publish(centers_msg)

            # Update last known pattern
            self.last_pattern = (error_distance, center_offset)

            # Publish debug image
            if self.pub_detection_image.get_num_connections() > 0:
                try:
                    debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                    self.pub_detection_image.publish(debug_msg)
                except Exception as e:
                    self.log(f"Error publishing debug image: {e}", 'error')

            return (error_distance, center_offset)
        else:
            # No pattern found
            cv2.putText(debug_img, "No pattern detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Publish debug image
            if self.pub_detection_image.get_num_connections() > 0:
                try:
                    debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                    self.pub_detection_image.publish(debug_msg)
                except Exception as e:
                    self.log(f"Error publishing debug image: {e}", 'error')

            return None

    def follow_pattern(self, error_distance, center_offset):
        """
        PID controller to follow the detected pattern

        Args:
            error_distance: Distance error (target - actual width)
            center_offset: Offset from center of image
        """
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.1  # Avoid division by zero

        # Compute velocity and omega based on error
        v = self.Kp_dist * error_distance
        omega = self.Kp_angle * center_offset

        # Calculate derivative term for smoother control
        d_offset = (center_offset - self.last_error) / dt
        omega += self.Kd * d_offset

        # Limit speed to avoid overshooting
        v = min(max(v, self.min_velocity), self.max_velocity) if v > 0 else 0

        # Create velocity command
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = v
        cmd.omega = omega

        # Publish command
        self.pub_cmd_vel.publish(cmd)

        # Update control state
        self.last_error = center_offset
        self.last_time = current_time

    def publish_detection_status(self, detected):
        """Publish detection status message"""
        detection_msg = BoolStamped()
        detection_msg.header.stamp = rospy.Time.now()
        detection_msg.data = detected
        self.pub_detection.publish(detection_msg)

    def stop(self):
        """Stop the robot"""
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = 0
        cmd.omega = 0
        self.pub_cmd_vel.publish(cmd)

    def on_shutdown(self):
        """Clean up when node is shut down"""
        # Stop the robot
        for i in range(5):  # Send multiple stop commands to ensure it's received
            self.stop()
        self.log("Circle Grid Follower Node shutting down")


if __name__ == '__main__':
    # Create the node
    node = CircleGridFollowerNode(node_name='circle_grid_follower_node')
    # Register shutdown hook
    rospy.on_shutdown(node.on_shutdown)
    # Keep the node running
    rospy.spin()
