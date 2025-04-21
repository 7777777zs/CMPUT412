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
from computer_vision.srv import SetLEDColor


class CombinedControllerNode(DTROS):
    """
    Combined Controller Node that handles both circle grid following and lane following.

    This node prioritizes grid following when a pattern is detected,
    and defaults to lane following when the grid is not visible.
    """

    def __init__(self, node_name):
        super(CombinedControllerNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION)

        # Grid position tracking for intersections
        self.grid_position = "center"  # Can be "left", "center", or "right"
        self.grid_position_history = []  # Store recent positions for stability
        self.position_history_length = 5  # Number of positions to keep in history

        # Add emergency stopping flag
        self.emergency_stopping = False

        # Add stop timing management
        self.stopping_duration = 3.0  # Duration to stop at red lines
        self.stop_start_time = 0
        self.stop_in_progress = False

        # Red intersection setup
        self.stopped_at_red = False
        self.time_of_red_stop = 0
        self.red_cooldown_duration = 10  # Seconds before detecting another red line
        self.red_stops_count = 0  # Counter for red stops
        self.is_turning_at_intersection = False
        self.turn_start_time = 0
        self.turn_duration = 1.5  # Duration for turning at intersection

        # Get vehicle name
        self.vehicle_name = os.environ.get('VEHICLE_NAME')
        if self.vehicle_name is None:
            raise ValueError("Environment variable VEHICLE_NAME is not set")

        # Wait for the service to be available before proceeding
        rospy.wait_for_service(f"/{self.vehicle_name}/set_led_color")
        self.set_led_color_service = rospy.ServiceProxy(
            f"/{self.vehicle_name}/set_led_color", SetLEDColor)

        # Operating mode
        self.mode = "lane_following"  # Start in lane following mode by default
        self.last_mode_switch = rospy.Time.now()
        # Replace with:
        # Number of consecutive missed frames before switching modes
        self.missed_frames_threshold = 3
        self.missed_frames_count = 0  # Counter for missed grid detections

        # Need to see grid for this many consecutive frames before switching back
        self.grid_found_frames_threshold = 3
        self.grid_found_frames_count = 0      # Counter for consecutive grid detections

        self.led_count = 0

        # Initialize bridge
        self.bridge = CvBridge()

        # ------------------------------------------------------------------------
        # Circle Grid Following Parameters
        # ------------------------------------------------------------------------

        # Process frequency
        self.process_frequency = rospy.get_param("~process_frequency", 10.0)
        self.circlepattern_dims = rospy.get_param(
            "~circlepattern_dims", [7, 3])

        # Detection parameters
        # Time before considering pattern lost
        self.detection_timeout = rospy.Duration(0.0)
        self.last_seen = rospy.Time(0)

        # Blob detector parameters
        self.blobdetector_min_area = rospy.get_param(
            "~blobdetector_min_area", 20)
        self.blobdetector_max_area = rospy.get_param(
            "~blobdetector_max_area", 2000)
        self.blobdetector_min_dist_between_blobs = rospy.get_param(
            "~blobdetector_min_dist_between_blobs", 5)

        # Grid following controller parameters (PID)
        self.grid_Kp_dist = rospy.get_param(
            "~grid_Kp_dist", 0.013)  # Proportional gain for distance
        # Proportional gain for angle/offset
        self.grid_Kp_angle = rospy.get_param("~grid_Kp_angle", -0.005)
        self.grid_Kd = rospy.get_param("~grid_Kd", 0.002)  # Derivative term

        # Pattern tracking variables
        self.last_pattern = (0, 0)  # (error_distance, center_offset)
        self.target_width = 100.0  # Target width for the pattern

        # Grid following controller state variables
        self.grid_last_error = 0
        self.grid_last_time = rospy.get_time()

        # Initialize timing variables
        self.last_stamp = rospy.Time.now()
        self.publish_duration = rospy.Duration.from_sec(
            1.0 / self.process_frequency)

        # Detection state
        self.pattern_detected = False

        self.min_velocity = 0.05
        self.max_velocity = 0.3

        # Setup blob detector
        self.setup_blob_detector()

        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ------------------------------------------------------------------------
        # Lane Following Parameters
        # ------------------------------------------------------------------------

        # Lane following controller parameters
        self.lane_kp = 0.035  # Proportional gain
        self.lane_kd = 0.01   # Derivative gain
        self.lane_ki = 0.001  # Integral gain

        # Lane controller state
        self.lane_error = 0
        self.lane_last_error = 0
        self.lane_integral = 0
        self.lane_derivative = 0
        self.lane_last_time = rospy.get_time()

        # Lane tracking parameters
        self.lane_width_pixels = 200  # Approximate lane width in pixels

        # Speed parameters
        self.normal_speed = 0.3      # Regular forward speed
        self.turn_speed = 0.3        # Speed during sharp turns
        self.current_speed = self.normal_speed

        # Image division parameters
        self.left_third_end = None   # Will be initialized when we get the first image
        self.right_third_start = None  # Will be initialized when we get the first image

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

        # Region of interest (ROI) for lane following
        self.roi_top = 250  # Top row of ROI
        self.roi_height = 120  # Height of ROI
        self.roi_width = 640  # Width of ROI (full width)

        # ------------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------------

        # Add this to your publishers section
        self.pub_red_line_vis = rospy.Publisher(
            f"/{self.vehicle_name}/red_line_detection/compressed",
            CompressedImage,
            queue_size=1
        )

        # Grid following publishers
        self.pub_grid_detection_image = rospy.Publisher(
            f"/{self.vehicle_name}/circle_grid_follower/detection_image/compressed",
            CompressedImage,
            queue_size=1
        )

        self.pub_grid_detection = rospy.Publisher(
            f"/{self.vehicle_name}/circle_grid_follower/detection",
            BoolStamped,
            queue_size=1
        )

        self.pub_centers = rospy.Publisher(
            f"/{self.vehicle_name}/circle_grid_follower/centers",
            VehicleCorners,
            queue_size=1
        )

        # Lane following publishers
        self.pub_lane_vis = rospy.Publisher(
            f"/{self.vehicle_name}/lane_following/image/compressed",
            CompressedImage,
            queue_size=1
        )

        # Common publishers
        self.pub_cmd_vel = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        self.pub_status = rospy.Publisher(
            f"/{self.vehicle_name}/combined_controller/status",
            String,
            queue_size=1
        )

        # ------------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------------

        # Camera subscriber
        self.img_sub = rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size="20MB"
        )

        # Timer for controller update (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        self.log("Combined Controller Node initialized")
        self.log(f"Starting in {self.mode} mode")

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

    def change_led_color(self, colors):
        """Change the LED colors using the service"""
        try:
            # Make sure colors is a list with exactly 5 elements
            if len(colors) != 5:
                rospy.logwarn(
                    f"Expected 5 colors, got {len(colors)}. Padding with 'off'")
                # Pad with 'off' if fewer than 5
                colors = colors + ['off'] * (5 - len(colors))
                # Or truncate if more than 5
                colors = colors[:5]

            # Call the service with the colors as a SINGLE argument
            result = self.set_led_color_service(colors)

            if result.success:
                rospy.loginfo("Successfully changed LED colors")
            else:
                rospy.logwarn("Failed to change LED colors")

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def update_grid_position(self, grid_center_x, img_width):
        """
        Update the tracked position of the grid in the image
        Args:
            grid_center_x: x-coordinate of the grid pattern center
            img_width: width of the image
        """
        # Determine which third of the image the grid is in
        left_third_end = img_width // 3
        right_third_start = (img_width * 2) // 3

        # Determine current position
        if grid_center_x < left_third_end:
            current_position = "left"
        elif grid_center_x > right_third_start:
            current_position = "right"
        else:
            current_position = "center"

        # Add to history
        self.grid_position_history.append(current_position)

        # Keep history at the desired length
        if len(self.grid_position_history) > self.position_history_length:
            self.grid_position_history.pop(0)

        # Update the current position based on majority vote in history
        if len(self.grid_position_history) > 0:
            left_count = self.grid_position_history.count("left")
            center_count = self.grid_position_history.count("center")
            right_count = self.grid_position_history.count("right")

            if left_count > center_count and left_count > right_count:
                self.grid_position = "left"
            elif right_count > center_count and right_count > left_count:
                self.grid_position = "right"
            else:
                self.grid_position = "center"

        rospy.loginfo(f"Grid position updated to: {self.grid_position}")

        return self.grid_position

    def image_callback(self, msg):
        """Process incoming camera images for both grid detection and lane following"""
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

        # Skip processing if currently executing a turn at an intersection
        if self.is_turning_at_intersection:
            current_time = rospy.get_time()
            if current_time - self.turn_start_time < self.turn_duration:
                # Still turning, do nothing else in this callback
                return
            else:
                # Turn completed
                self.is_turning_at_intersection = False
                rospy.loginfo("Intersection turn completed")

        # First, try to detect the circle grid
        detection_result = self.detect_circle_grid(image_cv)

        if detection_result is not None:
            # Reset missed frames counter when grid is detected
            self.missed_frames_count = 0

            # Grid detected - potentially switch to grid following mode
            if self.mode == "lane_following":
                self.change_led_color(
                    ['off', 'blue', 'off', 'off', 'blue'])

                # Increment the found frames counter
                self.grid_found_frames_count += 1
                self.log(
                    f"Grid found while lane following, found frames: {self.grid_found_frames_count}")

                # Only switch mode if we've seen the grid for enough consecutive frames
                if self.grid_found_frames_count >= self.grid_found_frames_threshold:
                    # When switching back to grid following:
                    # 1. First stop the robot
                    self.stop()

                    # 2. Add a brief pause to ensure the robot is fully stopped
                    rospy.loginfo(
                        "Stopping for safety before switching to grid following")
                    rospy.sleep(0.4)  # 200ms buffer

                    # 3. Send another stop command to be extra sure
                    self.stop()

                    # Now switch modes
                    self.mode = "grid_following"
                    self.last_mode_switch = now
                    self.log(
                        f"Grid found for {self.grid_found_frames_count} consecutive frames - switching to grid following mode")

                    # Reset the counter after switching
                    self.grid_found_frames_count = 0
            else:
                # Already in grid following mode, make sure counter is reset
                self.grid_found_frames_count = 0

            # Update pattern info regardless of mode
            error_distance, center_offset = detection_result
            self.pattern_detected = True
            self.last_seen = now
            self.last_pattern = (error_distance, center_offset)

            # Update the grid position tracking
            grid_center_x = image_cv.shape[1] // 2 + center_offset
            self.update_grid_position(grid_center_x, image_cv.shape[1])

            # Publish detection status
            self.publish_grid_detection_status(True)

            # Publish status
            status_msg = f"Grid detected: dist_error={error_distance:.2f}, offset={center_offset:.2f}, position={self.grid_position}"
            self.pub_status.publish(String(status_msg))
        else:
            self.grid_found_frames_count = 0
            # Grid not detected in current frame
            if self.mode == "grid_following":
                self.change_led_color(
                    ['off', 'green', 'off', 'off', 'green'])
                self.missed_frames_count += 1
                self.log(
                    f"Grid not found, missed frames: {self.missed_frames_count}")

                if self.missed_frames_count >= self.missed_frames_threshold:
                    # We've missed enough frames, switch to lane following
                    self.mode = "lane_following"
                    self.last_mode_switch = now
                    self.log(
                        f"Missed {self.missed_frames_count} frames - switching to lane following mode")
                    self.pattern_detected = False
                else:
                    # We can still use the last known pattern for a few frames
                    if self.pattern_detected:
                        error_distance, center_offset = self.last_pattern

                        # Publish detection status (still considering it detected until threshold)
                        self.publish_grid_detection_status(True)

                        # Publish status
                        status_msg = f"Using last grid pattern: dist_error={error_distance:.2f}, offset={center_offset:.2f}, missed frames: {self.missed_frames_count}, position={self.grid_position}"
                        self.pub_status.publish(String(status_msg))

        # Check for red line detection
        if self.mode == "lane_following":  # Only check for red lines when in lane-following mode
            # Check if we're in the cooldown period
            current_time = rospy.get_time()
            if (current_time - self.time_of_red_stop) > self.red_cooldown_duration:
                # Check for red line
                stopline_detected, area = self.detect_red_intersection(
                    image_cv)

                # Debug visualization if needed
                if stopline_detected:
                    # Use area as a better proximity measure - stop when area is large enough
                    stop_area_threshold = 2000  # Adjust based on testing
                    rospy.loginfo(f"Red line detected, area: {area:.2f}")

                    # Only stop if the area is large enough (meaning we're close enough)
                    if area > stop_area_threshold:
                        self.stop_at_red()
                        self.stopped_at_red = False  # Reset for next detection

                        self.publish_grid_detection_status(False)

                        # Process the image for lane following
                        rospy.loginfo("Calling process_lane_following")
                        self.process_lane_following(image_cv)
                        rospy.loginfo("Finished process_lane_following")

                        if self.stopped_at_red:
                            # We're in cooldown period but want to show we detected the line
                            self.pub_status.publish(String(
                                f"Red line cooldown: {int(self.red_cooldown_duration - (current_time - self.time_of_red_stop))}s remaining"))

    # ------------------------------------------------------------------------
    # Circle Grid Following Methods
    # ------------------------------------------------------------------------

    def detect_circle_grid(self, image_cv):
        """
        Detect circle grid pattern in the image without using any cropping

        Args:
            image_cv: OpenCV image

        Returns:
            tuple (error_distance, center_offset) or None if not detected
        """
        # Get image dimensions
        img_height, img_width = image_cv.shape[:2]

        # Calculate the boundaries for thirds
        left_third_end = img_width // 3
        right_third_start = (img_width * 2) // 3

        # Create debug image of full frame for visualization
        debug_img = image_cv.copy()

        # Draw vertical lines dividing the image into thirds
        cv2.line(debug_img, (left_third_end, 0),
                 (left_third_end, img_height), (0, 255, 0), 2)
        cv2.line(debug_img, (right_third_start, 0),
                 (right_third_start, img_height), (0, 255, 0), 2)

        # Add labels for the thirds
        cv2.putText(debug_img, "LEFT", (left_third_end // 2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, "CENTER", ((left_third_end + right_third_start) // 2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, "RIGHT", (right_third_start + (img_width - right_third_start) // 2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Process the full image without cropping
        process_img = image_cv

        # Add text indicating full image search is being used
        cv2.putText(debug_img, "Full Image Search", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Pre-process the image
        # 1) Convert to grayscale
        gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)

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

        if found:
            # Calculate pattern width and offset
            xs = centers[:, 0, 0]
            pattern_width = float(np.max(xs) - np.min(xs))
            error_distance = self.target_width - pattern_width  # Positive means too far

            # Calculate center offset
            center_offset = float(np.mean(xs) - img_width / 2)

            # Calculate the pattern's center x-coordinate
            pattern_center_x = float(np.mean(xs))

            # Determine which third the pattern is in
            pattern_position = "center"
            if pattern_center_x < left_third_end:
                pattern_position = "left"
            elif pattern_center_x > right_third_start:
                pattern_position = "right"

            # Update our tracking of grid position
            self.update_grid_position(pattern_center_x, img_width)

            # Draw detected pattern on debug image
            cv2.drawChessboardCorners(debug_img, tuple(
                self.circlepattern_dims), centers, found)

            # Add information text
            info_text = f"Grid: Width: {pattern_width:.1f}px, Offset: {center_offset:.1f}px"
            cv2.putText(debug_img, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add position information
            position_text = f"Position: {self.grid_position} (Current: {pattern_position})"
            cv2.putText(debug_img, position_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Publish centers if available
            if self.pub_centers.get_num_connections() > 0:
                centers_msg = VehicleCorners()
                centers_msg.header.stamp = rospy.Time.now()
                centers_msg.detection.data = True

                for point in centers:
                    center = Point32()
                    center.x = point[0][0]
                    center.y = point[0][1]
                    center.z = 0
                    centers_msg.corners.append(center)

                self.pub_centers.publish(centers_msg)

            # Update last known pattern
            self.last_pattern = (error_distance, center_offset)

            # Publish debug image
            if self.pub_grid_detection_image.get_num_connections() > 0:
                try:
                    debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                    self.pub_grid_detection_image.publish(debug_msg)
                except Exception as e:
                    self.log(
                        f"Error publishing grid debug image: {e}", 'error')

            return (error_distance, center_offset)
        else:
            # No pattern found
            cv2.putText(debug_img, "No grid pattern detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add current tracked position
            cv2.putText(debug_img, f"Last Position: {self.grid_position}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Publish debug image
            if self.pub_grid_detection_image.get_num_connections() > 0:
                try:
                    debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                    self.pub_grid_detection_image.publish(debug_msg)
                except Exception as e:
                    self.log(
                        f"Error publishing grid debug image: {e}", 'error')

            return None

    def follow_grid_pattern(self, error_distance, center_offset):
        """
        PID controller to follow the detected grid pattern

        Args:
            error_distance: Distance error (target - actual width)
            center_offset: Offset from center of image
        """

        current_time = rospy.get_time()
        dt = current_time - self.grid_last_time
        if dt <= 0:
            dt = 0.1  # Avoid division by zero

        # Compute velocity and omega based on error
        v = self.grid_Kp_dist * error_distance
        omega = self.grid_Kp_angle * center_offset

        # Calculate derivative term for smoother control
        d_offset = (center_offset - self.grid_last_error) / dt
        omega += self.grid_Kd * d_offset

        # Limit speed to avoid overshooting
        v = min(max(v, self.min_velocity), self.max_velocity) if v > 0 else 0

        # Debug output
        rospy.loginfo(
            f"Grid follow cmd: v={v:.3f}, omega={omega:.3f}, mode={self.mode}")

        # Create velocity command
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = v
        cmd.omega = omega

        # Publish command
        self.pub_cmd_vel.publish(cmd)

        # Update control state
        self.grid_last_error = center_offset
        self.grid_last_time = current_time

    def publish_grid_detection_status(self, detected):
        """Publish grid detection status message"""
        detection_msg = BoolStamped()
        detection_msg.header.stamp = rospy.Time.now()
        detection_msg.data = detected
        self.pub_grid_detection.publish(detection_msg)

    # ------------------------------------------------------------------------
    # Lane Following Methods
    # ------------------------------------------------------------------------

    def process_lane_following(self, img):
        """Process image for lane following"""
        try:
            # Crop to region of interest (lower part of the image)
            roi = img[self.roi_top:self.roi_top +
                      self.roi_height, 0:self.roi_width]

            # Calculate thirds if not already done
            if self.left_third_end is None:
                self.img_width = roi.shape[1]
                self.third_width = self.img_width // 3
                self.left_third_end = self.third_width
                self.right_third_start = self.third_width * 2
                self.log(
                    f"Image thirds: Left end={self.left_third_end}, Right start={self.right_third_start}")

            # Make a copy for visualization
            vis_img = roi.copy()

            # Detect lanes
            yellow_mask, yellow_center, yellow_contour = self.detect_lane(
                roi, 'yellow')
            white_mask, white_center, white_contour = self.detect_lane(
                roi, 'white')

            # Draw the vertical lines separating the thirds
            cv2.line(vis_img, (self.left_third_end, 0),
                     (self.left_third_end, roi.shape[0]), (0, 0, 255), 1)
            cv2.line(vis_img, (self.right_third_start, 0),
                     (self.right_third_start, roi.shape[0]), (0, 0, 255), 1)

            # Apply filtering logic based on image thirds
            # Only consider yellow line if it's in the left 2/3 of the image
            if yellow_center is not None and yellow_center > self.right_third_start:
                self.log(
                    "Yellow line detected in wrong position (right third), ignoring")
                yellow_center = None
                yellow_contour = None

            # Only consider white line if it's in the right 2/3 of the image
            if white_center is not None and white_center < self.left_third_end:
                self.log(
                    "White line detected in wrong position (left third), ignoring")
                white_center = None
                white_contour = None

            # Visualize lane detection
            if yellow_contour is not None:
                cv2.drawContours(
                    vis_img, [yellow_contour], -1, (0, 255, 255), 2)
                if yellow_center is not None:
                    cv2.circle(vis_img, (yellow_center,
                               roi.shape[0]//2), 5, (0, 255, 255), -1)

            if white_contour is not None:
                cv2.drawContours(
                    vis_img, [white_contour], -1, (255, 255, 255), 2)
                if white_center is not None:
                    cv2.circle(
                        vis_img, (white_center, roi.shape[0]//2), 5, (255, 255, 255), -1)

            # Calculate error based on lane detection
            self.calculate_lane_error(
                yellow_center, white_center, roi.shape[1])

            # Draw error and centerline visualizations
            self.draw_lane_error_visualization(vis_img)

            # Add mode information to the image
            cv2.putText(vis_img, "Lane Following Mode", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Add tracked grid position information
            cv2.putText(vis_img, f"Grid Position: {self.grid_position}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Publish visualization
            if self.pub_lane_vis.get_num_connections() > 0:
                vis_msg = self.bridge.cv2_to_compressed_imgmsg(vis_img)
                self.pub_lane_vis.publish(vis_msg)

        except Exception as e:
            self.log(f"Error processing lane following: {str(e)}", 'error')

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
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                cv2.line(img, (center_x, h//2),
                         (error_pos, h//2), (0, 255, 0), 2)

        # Label the thirds
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "L", (self.third_width // 2, 20),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "M", (self.third_width + self.third_width // 2, 20),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "R", (2 * self.third_width + self.third_width // 2, 20),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Add information about the error value
        if self.lane_error is not None:
            error_text = f"Error: {self.lane_error:.1f}px"
            cv2.putText(img, error_text, (10, h - 10),
                        font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        # Add current speed indicator
        speed_text = f"Speed: {self.current_speed:.2f}"
        cv2.putText(img, speed_text, (w - 120, h - 10),
                    font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        return img  # Return the modified image

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
            if lane_width < self.lane_width_pixels * 0.7:  # Lane is narrower than expected
                self.current_speed = self.turn_speed  # Slow down for turns
            else:
                self.current_speed = self.normal_speed  # Regular speed

            self.pub_status.publish(String(
                f"Lane following - both lines, center: {lane_center}, error: {self.lane_error}"))

        elif yellow_center is not None:
            # Only yellow lane detected - stay a fixed distance to the right
            # Offset by approx lane width/2
            self.lane_error = yellow_center - (img_center - 200)
            self.current_speed = self.turn_speed  # Slow down when only one lane is visible
            self.pub_status.publish(String(
                f"Lane following - yellow line, center: {yellow_center}, error: {self.lane_error}"))

        elif white_center is not None:
            # Only white lane detected - stay a fixed distance to the left
            # Offset by approx lane width/2
            self.lane_error = white_center - (img_center + 100)
            self.current_speed = self.turn_speed  # Slow down when only one lane is visible
            self.pub_status.publish(String(
                f"Lane following - white line, center: {white_center}, error: {self.lane_error}"))

        else:
            # No valid lane lines detected
            self.pub_status.publish(String(
                "Lane following - no lines detected, maintaining previous error"))
            # If no lanes detected, keep the previous error (no update)

    def control_loop(self, event):
        """Control loop that executes the appropriate controller based on current mode"""
        if self.emergency_stopping:
            return
        # Skip control if currently executing a turn at an intersection
        if self.is_turning_at_intersection:
            current_time = rospy.get_time()
            if current_time - self.turn_start_time < self.turn_duration:
                # Still turning, execute the turn
                self.execute_intersection_turn()
                return
            else:
                # Turn completed
                self.is_turning_at_intersection = False
                rospy.loginfo("Intersection turn completed")

        # Log the current state for debugging
        rospy.loginfo(
            f"Control loop running: mode={self.mode}, pattern_detected={self.pattern_detected}, lane_error={self.lane_error}")

        # Execute the appropriate controller based on current mode
        if self.mode == "grid_following" and self.pattern_detected:
            # Grid following mode with a pattern
            error_distance, center_offset = self.last_pattern
            rospy.loginfo(
                f"Grid following command will be issued: error_distance={error_distance}, center_offset={center_offset}")
            self.follow_grid_pattern(error_distance, center_offset)

            # Publish status for debugging
            status_msg = f"Grid following: dist_error={error_distance:.2f}, offset={center_offset:.2f}, position={self.grid_position}"
            self.pub_status.publish(String(status_msg))

        elif self.mode == "lane_following":
            # Lane following mode - call lane_control_loop
            rospy.loginfo(
                f"Lane following mode active, lane_error={self.lane_error}")

            # Only send commands if we have valid lane error
            if self.lane_error is not None:
                self.lane_control_loop()
                status_msg = f"Lane following - lane_error={self.lane_error:.2f}, speed={self.current_speed:.2f}, grid_position={self.grid_position}"
                self.pub_status.publish(String(status_msg))
            else:
                # If no lane detected, maintain a slow forward movement
                rospy.logwarn("No lane detected, moving slowly forward")
                cmd = Twist2DStamped()
                cmd.v = 0.1  # Very slow forward speed
                cmd.omega = 0  # No rotation
                self.pub_cmd_vel.publish(cmd)
                self.pub_status.publish(
                    String("Lane following - NO LANE DETECTED - Moving slowly forward"))

            self.publish_grid_detection_status(False)
        else:
            rospy.logwarn(
                f"No control action taken: mode={self.mode}, pattern_detected={self.pattern_detected}")
            # If we're in an unexpected state, stop the robot for safety
            self.stop()

    def lane_control_loop(self):
        """PID control loop for lane following"""
        if self.lane_error is None:
            rospy.logwarn("Lane control loop called with lane_error=None")
            return

        # Calculate dt
        current_time = rospy.get_time()
        dt = current_time - self.lane_last_time
        if dt <= 0:
            dt = 0.1  # Avoid division by zero or negative time
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
        omega = -self.lane_kp * self.lane_error - self.lane_kd * \
            self.lane_derivative - self.lane_ki * self.lane_integral

        # Limit the maximum omega to prevent excessive rotation
        max_omega = 8.0
        omega = max(min(omega, max_omega), -max_omega)

        # Create and publish velocity command
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = self.current_speed  # Use adaptive speed
        cmd.omega = omega
        self.pub_cmd_vel.publish(cmd)

        # Log the command for debugging
        rospy.loginfo(
            f"Lane control: v={self.current_speed:.3f}, omega={omega:.3f}, error={self.lane_error:.2f}")

        # Save error for next iteration
        self.lane_last_error = self.lane_error

        # Publish status for debugging/monitoring
        status_msg = (f"Lane following - Error: {self.lane_error:.2f}, D: {self.lane_derivative:.2f}, "
                      f"I: {self.lane_integral:.2f}, Omega: {omega:.2f}, Speed: {self.current_speed:.2f}")
        self.pub_status.publish(String(status_msg))

    def detect_red_intersection(self, image):
        """
        Detect red intersection lines in the image, cropping the top 1/3
        Returns: (detected, distance) where:
            - detected: Boolean indicating if a red line is detected
            - distance: Estimated distance to the line (or infinity if none detected)
        """
        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Crop the bottom 2/3 of the image (remove top 1/3)
        crop_top = int(img_height * 1/3)
        cropped_image = image[crop_top:, :]

        # Create a copy for visualization
        vis_img = cropped_image.copy()

        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # Define red color range
        red_ranges = {'lower': np.array(
            [0, 150, 50]), 'upper': np.array([10, 255, 255])}

        # Create mask for red color
        mask = cv2.inRange(hsv, red_ranges['lower'], red_ranges['upper'])

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours to find a red line
        detected = False
        distance = float('inf')

        # Minimum area to consider for detection (in pixels)
        min_area_threshold = 500

        # Area threshold for when to actually stop (larger area = closer)
        stop_area_threshold = 2000  # Adjust this value based on testing

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Check if contour is large enough to be a line
            if area > min_area_threshold:
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Draw rectangle around detected line
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Draw contour
                cv2.drawContours(
                    vis_img, [largest_contour], -1, (0, 255, 255), 2)

                # Use the area as the "distance" measure (inverse relationship)
                # Larger area = closer to the line
                distance = area

                # Add text with area info
                cv2.putText(vis_img, f"Red Line Area: {area:.0f}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add area info
                cv2.putText(vis_img, f"Stop Threshold: {stop_area_threshold}px", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add tracked grid position
                cv2.putText(vis_img, f"Grid Position: {self.grid_position}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                detected = True

                # Visualize the mask as an overlay
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(vis_img, 0.7, mask_colored, 0.3, 0)

                # Publish visualization
                if self.pub_red_line_vis.get_num_connections() > 0:
                    try:
                        # Create a full-sized visualization image with the original dimensions
                        full_vis = image.copy()
                        # Mark the cropping boundary with a horizontal line
                        cv2.line(full_vis, (0, crop_top),
                                 (img_width, crop_top), (0, 255, 0), 2)
                        # Place the processed image in the cropped region
                        full_vis[crop_top:, :] = overlay
                        # Add text to indicate cropping
                        cv2.putText(full_vis, "Cropped region for red line detection",
                                    (10, crop_top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)

                        vis_msg = self.bridge.cv2_to_compressed_imgmsg(
                            full_vis)
                        self.pub_red_line_vis.publish(vis_msg)
                    except Exception as e:
                        self.log(
                            f"Error publishing red line visualization: {e}", 'error')
            else:
                # No large enough contour
                cv2.putText(vis_img, "No red line detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Publish visualization anyway to show what's being seen
                if self.pub_red_line_vis.get_num_connections() > 0:
                    try:
                        # Create a full-sized visualization image
                        full_vis = image.copy()
                        # Mark the cropping boundary
                        cv2.line(full_vis, (0, crop_top),
                                 (img_width, crop_top), (0, 255, 0), 2)
                        # Place the processed image in the cropped region
                        full_vis[crop_top:, :] = vis_img
                        # Add text
                        cv2.putText(full_vis, "Cropped region for red line detection",
                                    (10, crop_top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)

                        vis_msg = self.bridge.cv2_to_compressed_imgmsg(
                            full_vis)
                        self.pub_red_line_vis.publish(vis_msg)
                    except Exception as e:
                        self.log(
                            f"Error publishing red line visualization: {e}", 'error')
        else:
            # No contours at all
            cv2.putText(vis_img, "No red line detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Publish visualization anyway
            if self.pub_red_line_vis.get_num_connections() > 0:
                try:
                    # Create a full-sized visualization image
                    full_vis = image.copy()
                    # Mark the cropping boundary
                    cv2.line(full_vis, (0, crop_top),
                             (img_width, crop_top), (0, 255, 0), 2)
                    # Place the processed image in the cropped region
                    full_vis[crop_top:, :] = vis_img
                    # Add text
                    cv2.putText(full_vis, "Cropped region for red line detection",
                                (10, crop_top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

                    vis_msg = self.bridge.cv2_to_compressed_imgmsg(full_vis)
                    self.pub_red_line_vis.publish(vis_msg)
                except Exception as e:
                    self.log(
                        f"Error publishing red line visualization: {e}", 'error')

        return detected, distance if detected else float('inf')

    def stop_at_red(self):
        """Handle the process of stopping at a red line and then turn based on grid position"""
        if not self.stopped_at_red:
            rospy.loginfo(
                f"Stopping at red line (stop #{self.red_stops_count + 1})")
            self.time_of_red_stop = rospy.get_time()

            # Start emergency stop
            self.emergency_stopping = True
            self.stop_in_progress = True
            self.stop_start_time = rospy.get_time()

            # Stop the robot without allowing other commands to interfere
            self.emergency_stop()

            # Schedule a callback to execute after stop duration
            rospy.Timer(rospy.Duration(self.stopping_duration),
                        self.handle_post_stop_actions, oneshot=True)

            self.stopped_at_red = True
            self.red_stops_count += 1

            # Publish status message
            status_msg = f"Stopped at red line - stop #{self.red_stops_count}"
            self.pub_status.publish(String(status_msg))

    def execute_intersection_turn(self, direction="straight"):
        """Execute a turn at an intersection based on the specified direction"""
        # Ensure we have a valid publisher
        if self.pub_cmd_vel is None or not hasattr(self.pub_cmd_vel, 'publish'):
            rospy.logwarn(
                "Command velocity publisher not valid during turn, recreating...")
            self.pub_cmd_vel = rospy.Publisher(
                f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
                Twist2DStamped,
                queue_size=1
            )
            rospy.sleep(0.1)  # Small delay to let publisher initialize

        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        self.change_led_color(['off', 'blue', 'off', 'off', 'blue'])

        if direction == "left":
            cmd.v = 0.2  # Slow forward speed during turn
            cmd.omega = 3.0  # Strong left turn

        elif direction == "right":
            cmd.v = 0.2  # Slow forward speed during turn
            cmd.omega = -3.0  # Strong right turn

        else:  # straight
            cmd.v = 0.2  # Medium forward speed
            cmd.omega = 0.0  # No turning

        try:
            self.pub_cmd_vel.publish(cmd)
            rospy.loginfo(
                f"Executing {direction} turn at intersection: v={cmd.v}, omega={cmd.omega}")
        except Exception as e:
            rospy.logerr(f"Error publishing turn command: {e}")
            # If publishing fails, try to recreate the publisher
            self.pub_cmd_vel = rospy.Publisher(
                f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
                Twist2DStamped,
                queue_size=1
            )
            rospy.sleep(0.2)  # Give it a bit more time to initialize
            try:
                self.pub_cmd_vel.publish(cmd)
                rospy.loginfo(
                    f"Retry: Executing {direction} turn at intersection")
            except Exception as e2:
                rospy.logerr(
                    f"Second attempt to publish turn command failed: {e2}")

    def handle_post_stop_actions(self, event):
        """Actions to perform after stopping duration is complete"""
        if not self.stop_in_progress:
            return

        # Reset stopping flags
        self.stop_in_progress = False
        self.emergency_stopping = False

        # Make sure we have a valid publisher
        if self.pub_cmd_vel is None or not hasattr(self.pub_cmd_vel, 'publish'):
            rospy.logwarn(
                "Command velocity publisher not valid, recreating...")
            self.pub_cmd_vel = rospy.Publisher(
                f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
                Twist2DStamped,
                queue_size=1
            )
            rospy.sleep(0.1)  # Small delay to let publisher initialize

        # Decide which way to turn based on grid position
        self.is_turning_at_intersection = True
        self.turn_start_time = rospy.get_time()

        turn_direction = "straight"
        if self.grid_position == "left":
            turn_direction = "left"
            rospy.loginfo("Grid was on LEFT - turning LEFT at intersection")
        elif self.grid_position == "right":
            turn_direction = "right"
            rospy.loginfo("Grid was on RIGHT - turning RIGHT at intersection")
        else:
            rospy.loginfo(
                "Grid was in CENTER - going STRAIGHT at intersection")

        # Execute the turn
        self.execute_intersection_turn(turn_direction)

    def stop(self):
        self.emergency_stop()

    def emergency_stop(self):
        """Emergency stop that prevents other commands from being processed"""
        # Create a new stop command
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = 0
        cmd.omega = 0

        # Store the original publisher
        original_publisher = self.pub_cmd_vel

        try:
            # Send multiple stop commands with the current publisher
            for i in range(5):
                self.pub_cmd_vel.publish(cmd)
                rospy.sleep(0.05)  # Small delay between stop commands

            # Only unregister if we're planning to create a new one
            if original_publisher:
                original_publisher.unregister()
                rospy.sleep(0.1)  # Brief pause to allow unregistration

                # Create a new publisher with same topic
                self.pub_cmd_vel = rospy.Publisher(
                    f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
                    Twist2DStamped,
                    queue_size=1
                )
                rospy.sleep(0.1)  # Wait for publisher to initialize

                # Send one more stop command with the new publisher
                self.pub_cmd_vel.publish(cmd)
        except Exception as e:
            rospy.logerr(f"Error in emergency stop: {e}")
            # If there was an error, try to restore the original publisher
            if original_publisher and original_publisher != self.pub_cmd_vel:
                self.pub_cmd_vel = original_publisher

        rospy.loginfo(
            "EMERGENCY STOP command issued - vehicle should now be stopped")

    def on_shutdown(self):
        """Clean up when node is shut down"""
        # Use emergency stop to ensure robot really stops
        self.emergency_stopping = True
        self.emergency_stop()
        self.log("Combined Controller Node shutting down")


if __name__ == '__main__':
    # Create the node
    node = CombinedControllerNode(node_name='combined_controller_node')
    # Register shutdown hook
    rospy.on_shutdown(node.on_shutdown)
    # Keep the node running
    rospy.spin()
