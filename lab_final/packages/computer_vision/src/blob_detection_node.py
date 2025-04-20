#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import BoolStamped, VehicleCorners, LEDPattern
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA, Float32


class OptimizedBlobDetectionNode(DTROS):
    """
    Optimized blob detection node that detects checkerboard patterns.

    This node focuses on efficient detection by prioritizing the method that works well
    for the specific environment.
    """

    def __init__(self, node_name):
        super(OptimizedBlobDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION)

        # Get vehicle name from environment variable
        self.vehicle_name = os.environ.get('VEHICLE_NAME')
        if self.vehicle_name is None:
            raise ValueError("Environment variable VEHICLE_NAME is not set")

        # Parameters for blob detection
        self.process_frequency = rospy.get_param(
            "~process_frequency", 10.0)  # Increased from 2.0
        self.circlepattern_dims = rospy.get_param(
            "~circlepattern_dims", [7, 3])

        # Parameters for blob detection
        self.blobdetector_min_area = rospy.get_param(
            "~blobdetector_min_area", 20)
        self.blobdetector_max_area = rospy.get_param(
            "~blobdetector_max_area", 2000)
        self.blobdetector_min_dist_between_blobs = rospy.get_param(
            "~blobdetector_min_dist_between_blobs", 5)

        # Pre-processing parameters
        self.apply_erosion = rospy.get_param("~apply_erosion", True)
        self.erosion_kernel_size = rospy.get_param("~erosion_kernel_size", 3)
        self.erosion_iterations = rospy.get_param("~erosion_iterations", 1)

        # Parameters for advanced detection
        self.adaptive_mode = rospy.get_param("~adaptive_mode", True)
        self.threshold_block_size = rospy.get_param(
            "~threshold_block_size", 21)
        self.threshold_constant = rospy.get_param("~threshold_constant", 2)

        # Multiscale detection
        self.enable_multiscale = rospy.get_param(
            "~enable_multiscale", False)
        self.scale_factors = rospy.get_param("~scale_factors", [1.0, 0.75])

        # Homography matrix for distance calculation
        self.homography_param = f"/{self.vehicle_name}/updated_extrinsics"
        self.homography = np.array(rospy.get_param(
            self.homography_param, default=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])).reshape(3, 3)

        # Debug mode and verbose logging
        self.debug_mode = rospy.get_param(
            "~debug_mode", False)  # Disabled by default
        self.verbose_logging = rospy.get_param(
            "~verbose_logging", False)  # Less verbose by default

        # Topic name of the corrected image
        self.corrected_img_topic = rospy.get_param(
            "~corrected_img_topic",
            f"/{self.vehicle_name}/undistorted_node/image/compressed"
        )

        # Initialize timing variables
        self.last_stamp = rospy.Time.now()
        self.publish_duration = rospy.Duration.from_sec(
            1.0 / self.process_frequency)

        # Distance tracking
        self.board_distance = -1.0  # Distance to checkerboard, -1 if not detected

        # Success counters for adaptive strategy
        self.original_success_count = 0
        self.multiscale_success_count = 0
        self.total_frames = 0
        self.adaptation_interval = 20  # Check strategy effectiveness every N frames

        # Counter for first callback
        self.callback_counter = 0
        self.leds_turned_off = False

        # Initialize blob detector
        self.setup_blob_detector()

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Publishers for blob detection
        self.pub_centers = rospy.Publisher(
            f"/{self.vehicle_name}/blob_detection_node/centers",
            VehicleCorners,
            queue_size=1
        )
        self.pub_detection_image = rospy.Publisher(
            f"/{self.vehicle_name}/blob_detection_node/detection_image/compressed",
            CompressedImage,
            queue_size=1
        )
        self.pub_detection = rospy.Publisher(
            f"/{self.vehicle_name}/blob_detection_node/detection",
            BoolStamped,
            queue_size=1
        )

        # Publishers for distance information
        self.pub_board_distance = rospy.Publisher(
            f"/{self.vehicle_name}/blob_detection_node/board_distance",
            Float32,
            queue_size=1
        )

        # Debug publishers
        if self.debug_mode:
            self.pub_processed_image = rospy.Publisher(
                f"/{self.vehicle_name}/blob_detection_node/processed_image/compressed",
                CompressedImage,
                queue_size=1
            )

        # LED pattern publisher
        self.led_pattern_pub = rospy.Publisher(
            f"/{self.vehicle_name}/led_emitter_node/led_pattern",
            LEDPattern,
            queue_size=1
        )

        # Subscribe to the corrected image topic
        self.img_sub = rospy.Subscriber(
            self.corrected_img_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=10000000
        )

        self.log(f"Optimized Blob Detection Node initialized successfully")
        self.log(
            f"Subscribed to corrected image topic: {self.corrected_img_topic}")
        self.log(f"Front LEDs will be turned off on first image received")
        self.log(f"Based on logs, prioritizing original image detection method")

    def setup_blob_detector(self):
        """
        Configure and create a blob detector optimized for large and close blobs.
        """
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds - wider range for better adaptability
        params.minThreshold = 10
        params.maxThreshold = 220
        params.thresholdStep = 10

        # Filter by Area - increased max area for larger, closer blobs
        params.filterByArea = True
        params.minArea = self.blobdetector_min_area
        params.maxArea = self.blobdetector_max_area

        # Less strict circularity for potentially merged blobs
        params.filterByCircularity = True
        params.minCircularity = 0.5  # Reduced for larger blobs

        # Reduced convexity requirement for merged blobs
        params.filterByConvexity = True
        params.minConvexity = 0.6  # Reduced for larger blobs

        # Less strict inertia ratio for potentially elongated merged blobs
        params.filterByInertia = True
        params.minInertiaRatio = 0.3  # Reduced for larger blobs

        # Minimum distance between blobs - reduced for close blobs
        params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs

        # Create detector
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)

    def turn_off_front_leds(self):
        """Turn off all LEDs"""
        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()

        # Create LED pattern with all LEDs off
        rgb_vals = []

        for i in range(5):
            color_rgba = ColorRGBA()
            # Turn off (black)
            color_rgba.r = 0.0
            color_rgba.g = 0.0
            color_rgba.b = 0.0

            color_rgba.a = 1.0
            rgb_vals.append(color_rgba)

        pattern_msg.rgb_vals = rgb_vals
        pattern_msg.frequency = 0.0

        self.led_pattern_pub.publish(pattern_msg)

        # Mark that we've turned off the LEDs
        self.leds_turned_off = True
        self.log("All LEDs turned off")

    def preprocess_image(self, image_cv):
        """
        Apply specialized preprocessing for large/close blob detection.

        Args:
            image_cv (np.ndarray): Input image

        Returns:
            tuple: (preprocessed_image, debug_image)
        """
        # Convert to grayscale if needed
        if len(image_cv.shape) == 3:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_cv

        # Create a debug image if in debug mode
        debug_img = None
        if self.debug_mode:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding to better separate close blobs
        if self.adaptive_mode:
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.threshold_block_size,
                self.threshold_constant
            )
        else:
            # Fallback to standard thresholding
            _, binary = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Apply erosion to separate close blobs if enabled
        if self.apply_erosion:
            kernel = np.ones(
                (self.erosion_kernel_size, self.erosion_kernel_size), np.uint8)
            binary = cv2.erode(
                binary, kernel, iterations=self.erosion_iterations)

        # Update debug image if requested
        if debug_img is not None:
            # Add the processed binary image as an overlay
            overlay = debug_img.copy()
            mask_color = np.zeros_like(debug_img)
            # Green overlay for detected areas
            mask_color[binary > 0] = [0, 255, 0]
            cv2.addWeighted(mask_color, 0.3, overlay, 0.7, 0, debug_img)

        return binary, debug_img

    def get_distance_to_point(self, u, v):
        """
        Calculate the distance to point in world coordinates using homography

        Args:
            u, v: Pixel coordinates

        Returns:
            tuple: (X, Y, euclidean_distance)
        """
        pixel_coords = np.array([u, v, 1]).reshape(3, 1)
        world_coords = np.dot(self.homography, pixel_coords)

        # Convert from homogeneous coordinates to (X, Y)
        X = world_coords[0] / world_coords[2]
        Y = world_coords[1] / world_coords[2]
        euclidean_dist = (X**2 + Y**2)**(1/2)

        # Ensure we return float values
        if isinstance(euclidean_dist, np.ndarray):
            euclidean_dist = float(euclidean_dist)

        return float(X), float(Y), euclidean_dist

    def calculate_board_distance(self, centers):
        """
        Calculate the distance to the detected checkerboard

        Args:
            centers: Centers of detected blobs

        Returns:
            float: Distance to the board
        """
        if centers is None or len(centers) == 0:
            return -1.0

        # Calculate centroid of all blob centers
        center_points = [p[0] for p in centers]
        center_x = np.mean([p[0] for p in center_points])
        center_y = np.mean([p[1] for p in center_points])

        # Calculate distance using homography
        _, _, distance = self.get_distance_to_point(center_x, center_y)

        return distance

    def image_callback(self, image_msg):
        """
        Process incoming corrected camera images to detect blobs.

        Args:
            image_msg (CompressedImage): Incoming corrected/undistorted camera image
        """
        # Turn off LEDs on first callback
        if self.callback_counter == 0:
            self.turn_off_front_leds()

        # Increment callback counter
        self.callback_counter += 1

        now = rospy.Time.now()

        # Rate limiting
        if now - self.last_stamp < self.publish_duration:
            return

        self.last_stamp = now
        self.total_frames += 1

        # Convert compressed image to OpenCV format
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.log(f"Error converting image: {e}", 'error')
            return

        # Reset distance value
        self.board_distance = -1.0

        # Optimized detection strategy - try original image first since it works well
        detection = False
        centers = None

        # Try with original image first - most efficient method
        detection, centers = self.detect_blobs_original(image_cv)

        if detection:
            self.original_success_count += 1
            if self.verbose_logging:
                self.log("Detected board with original method")
        elif self.enable_multiscale:  # Only try multi-scale if enabled and needed
            # Try multi-scale detection as fallback
            for scale in self.scale_factors:
                if scale != 1.0:
                    # Use preprocessing only for multi-scale approach to save computation
                    processed_img, _ = self.preprocess_image(image_cv)
                    h, w = processed_img.shape[:2]
                    scaled_img = cv2.resize(
                        processed_img, (int(w*scale), int(h*scale)))
                    detection, scaled_centers = self.detect_blobs(scaled_img)

                    # If detection successful, scale centers back to original size
                    if detection:
                        centers = scaled_centers.copy()
                        # Scale coordinates back
                        for i in range(len(centers)):
                            centers[i][0][0] /= scale
                            centers[i][0][1] /= scale
                        self.multiscale_success_count += 1
                        if self.verbose_logging:
                            self.log(f"Detected board at scale {scale}")
                        break

        # Calculate board distance if detected
        if detection:
            self.board_distance = self.calculate_board_distance(centers)
            if self.verbose_logging:
                self.log(
                    f"Board detected at distance: {self.board_distance:.3f}m")
        elif self.verbose_logging:
            self.log("No board detected in image")

        # Publish distance information
        self.pub_board_distance.publish(Float32(data=self.board_distance))

        # Publish detection status
        detection_msg = BoolStamped()
        detection_msg.header.stamp = now
        detection_msg.data = detection
        self.pub_detection.publish(detection_msg)

        # Publish centers if blobs were detected
        if detection:
            centers_msg = VehicleCorners()
            centers_msg.header.stamp = now
            centers_msg.detection.data = True

            # Convert centers to Point32 messages
            for point in centers:
                center = Point32()
                center.x = point[0, 0]
                center.y = point[0, 1]
                center.z = 0
                centers_msg.corners.append(center)

            self.pub_centers.publish(centers_msg)

        # Publish visualization image only if subscribers exist
        if self.pub_detection_image.get_num_connections() > 0:
            annotated_img = image_cv.copy()

            # Add distance information text
            cv2.putText(annotated_img, f"Board dist: {self.board_distance:.3f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if detection else (0, 0, 255), 2)

            if detection:
                cv2.drawChessboardCorners(
                    annotated_img,
                    tuple(self.circlepattern_dims),
                    centers,
                    detection
                )

            try:
                image_msg_out = self.bridge.cv2_to_compressed_imgmsg(
                    annotated_img)
                image_msg_out.header = image_msg.header
                self.pub_detection_image.publish(image_msg_out)
            except Exception as e:
                self.log(f"Error converting annotated image: {e}", 'error')

        # Publish debug image if in debug mode and subscribers exist
        if self.debug_mode and self.pub_processed_image.get_num_connections() > 0:
            _, debug_img = self.preprocess_image(image_cv)
            if debug_img is not None:
                try:
                    debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
                    debug_msg.header = image_msg.header
                    self.pub_processed_image.publish(debug_msg)
                except Exception as e:
                    self.log(f"Error publishing debug image: {e}", 'error')

        # Adapt strategy periodically
        if self.total_frames % self.adaptation_interval == 0:
            self.adapt_detection_strategy()

    def adapt_detection_strategy(self):
        """Adapt detection strategy based on success rates"""
        if self.total_frames < self.adaptation_interval:
            return

        original_success_rate = self.original_success_count / self.total_frames
        multiscale_success_rate = self.multiscale_success_count / self.total_frames

        # Enable/disable multi-scale based on success rates
        if multiscale_success_rate > 0.2 and multiscale_success_rate > original_success_rate:
            if not self.enable_multiscale:
                self.log("Enabling multi-scale detection based on success rates")
                self.enable_multiscale = True
        elif original_success_rate > 0.6 and self.enable_multiscale:
            self.log(
                "Disabling multi-scale detection as original method is sufficient")
            self.enable_multiscale = False

        # Log current statistics
        if self.verbose_logging:
            self.log(
                f"Detection statistics - Original: {original_success_rate:.2f}, Multi-scale: {multiscale_success_rate:.2f}")

    def detect_blobs(self, image):
        """
        Detect circle grid in preprocessed image.

        Args:
            image (np.ndarray): Preprocessed image

        Returns:
            tuple: (detection status, centers of detected blobs)
        """
        # Find circle grid
        (detection, centers) = cv2.findCirclesGrid(
            image,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.simple_blob_detector,
        )

        return detection, centers

    def detect_blobs_original(self, image):
        """
        Detection on original image.

        Args:
            image (np.ndarray): Original color image

        Returns:
            tuple: (detection status, centers of detected blobs)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find circle grid with standard approach
        (detection, centers) = cv2.findCirclesGrid(
            gray,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.simple_blob_detector,
        )

        return detection, centers

    def on_shutdown(self):
        """Handle shutdown gracefully"""
        self.log("Shutting down, LED state remains off")


if __name__ == '__main__':
    # Create and start the node
    node = OptimizedBlobDetectionNode(
        node_name='optimized_blob_detection_node')
    # Register shutdown hook
    rospy.on_shutdown(node.on_shutdown)
    # Keep the node running
    rospy.spin()
