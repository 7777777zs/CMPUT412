#!/usr/bin/env python3

import rospy
import numpy as np
from os import uname
from re import match
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge


class BlueDetectionNode(DTROS):
    def __init__(self, node_name):
        super(BlueDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname

        # Image processing topics
        self._undistort_topic = f"/{self._vehicle_name}/undistorted_node/image/compressed"
        self._colour_detection_topic = f"/{self._vehicle_name}/blue_detection_node/image/compressed"
        self._homography_param = f"/{self._vehicle_name}/updated_extrinsics"

        # Publishers and subscribers
        self.sub = rospy.Subscriber(
            self._undistort_topic, CompressedImage, self.camera_reader_callback, queue_size=1)

        # Publisher for visualization
        self.img_pub = rospy.Publisher(
            self._colour_detection_topic, CompressedImage, queue_size=1)

        # Publisher for the nearest blue line distance
        self.distance_pub = rospy.Publisher(
            f"/{self._vehicle_name}/blue_line_distance", Float32, queue_size=1)

        # Publisher for blue line double detection (publishes only when two blue lines are detected)
        self.blue_double_pub = rospy.Publisher(
            f"/{self._vehicle_name}/blue_line_double_distance", Float32, queue_size=1)

        self._bridge = CvBridge()
        self._homography = np.array(rospy.get_param(
            self._homography_param, default=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])).reshape(3, 3)

        # Kernel for morphological operations
        self.kernel = np.ones((3, 3), "uint8")

        # Blue color range in HSV
        # OpenCV HSV: H is 0-179, S is 0-255, V is 0-255
        self.blue_lower = (100, 100, 100)
        self.blue_upper = (140, 255, 255)

        # Minimum area for valid blue line detection
        self.min_contour_area = 300

        # Minimum distance between two blue lines to be considered separate
        self.min_line_separation = 0.05  # in world units

        rospy.loginfo("Blue detection node initialized")

    def detect_blue(self, hsvFrame):
        """Detect blue color in HSV image"""
        mask = cv2.inRange(hsvFrame, np.array(
            self.blue_lower, dtype=np.uint8), np.array(self.blue_upper, dtype=np.uint8))
        mask = cv2.dilate(mask, self.kernel)
        return mask

    def get_distance_to_point(self, u, v):
        """Calculate the distance to point in world coordinates using homography"""
        pixel_coords = np.array([u, v, 1]).reshape(3, 1)
        world_coords = np.dot(self._homography, pixel_coords)

        # Convert from homogeneous coordinates to (X, Y)
        X = world_coords[0] / world_coords[2]
        Y = world_coords[1] / world_coords[2]
        euclidean_dist = (X**2 + Y**2)**(1/2)

        return X, Y, euclidean_dist

    def calculate_world_distance(self, point1, point2):
        """Calculate distance between two points in world coordinates"""
        x1, y1, _ = self.get_distance_to_point(point1[0], point1[1])
        x2, y2, _ = self.get_distance_to_point(point2[0], point2[1])

        # Ensure we're working with Python float values
        if isinstance(x1, np.ndarray):
            x1 = float(x1)
        if isinstance(y1, np.ndarray):
            y1 = float(y1)
        if isinstance(x2, np.ndarray):
            x2 = float(x2)
        if isinstance(y2, np.ndarray):
            y2 = float(y2)

        # Calculate Euclidean distance in world coordinates
        return float(np.sqrt((x2-x1)**2 + (y2-y1)**2))

    def camera_reader_callback(self, msg):
        if rospy.is_shutdown():
            return

        # Convert the compressed image to an OpenCV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Convert the image to HSV
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a copy for visualization
        output_image = image.copy()

        # Detect blue
        mask = self.detect_blue(hsvFrame)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize distances
        closest_distance = float('inf')
        double_line_distance = float('inf')

        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(
            c) > self.min_contour_area]

        # Sort contours by area in descending order
        valid_contours.sort(key=cv2.contourArea, reverse=True)

        # We need at least two blue lines
        if len(valid_contours) >= 2:
            # Get the two largest contours
            contour1 = valid_contours[0]
            contour2 = valid_contours[1]

            # Get bounding rectangles
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)

            # Calculate centers
            center1 = (x1 + w1 // 2, y1 + h1 // 2)
            center2 = (x2 + w2 // 2, y2 + h2 // 2)

            # Calculate distance between the two blue lines in world coordinates
            line_separation = self.calculate_world_distance(center1, center2)

            # Check if the lines are sufficiently separated
            if line_separation >= self.min_line_separation:
                # Draw the detected blue lines
                cv2.rectangle(output_image, (x1, y1),
                              (x1 + w1, y1 + h1), (255, 0, 0), 2)
                cv2.rectangle(output_image, (x2, y2),
                              (x2 + w2, y2 + h2), (255, 0, 0), 2)

                # Draw center points
                cv2.circle(output_image, center1, 5,
                           (0, 255, 255), -1)  # Yellow dot
                cv2.circle(output_image, center2, 5,
                           (0, 255, 255), -1)  # Yellow dot

                # Calculate distances to each blue line
                _, _, dist1 = self.get_distance_to_point(
                    center1[0], center1[1])
                _, _, dist2 = self.get_distance_to_point(
                    center2[0], center2[1])

                # Ensure distances are floats
                if isinstance(dist1, np.ndarray):
                    dist1 = float(dist1)
                if isinstance(dist2, np.ndarray):
                    dist2 = float(dist2)

                # Find the closer line
                if dist1 <= dist2:
                    closest_distance = dist1
                    closest_center = center1
                    label_text = "Near"
                    other_center = center2
                    other_text = "Far"
                else:
                    closest_distance = dist2
                    closest_center = center2
                    label_text = "Near"
                    other_center = center1
                    other_text = "Far"

                # Set the double line distance to the distance to the closer line
                double_line_distance = closest_distance

                # Add text showing the distances and labels
                cv2.putText(output_image, f"{label_text}: {closest_distance:.3f}m",
                            (closest_center[0] - 70, closest_center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.putText(output_image, f"{other_text}: {max(dist1, dist2):.3f}m",
                            (other_center[0] - 70, other_center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Show the separation distance
                mid_x = (center1[0] + center2[0]) // 2
                mid_y = (center1[1] + center2[1]) // 2
                cv2.putText(output_image, f"Sep: {line_separation:.3f}m",
                            (mid_x - 70, mid_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw a line connecting the two centers
                cv2.line(output_image, center1, center2, (0, 255, 255), 2)

                # Publish double blue line detection
                self.blue_double_pub.publish(Float32(double_line_distance))

                # Also publish the closest blue line distance
                self.distance_pub.publish(Float32(closest_distance))

                rospy.loginfo_throttle(1.0,
                                       f"Detected two blue lines, separation: {line_separation:.2f}m, closest: {closest_distance:.2f}m")
            else:
                # If lines aren't sufficiently separated, publish infinity
                self.blue_double_pub.publish(Float32(float('inf')))

                # Still draw the contours but with different color
                cv2.rectangle(output_image, (x1, y1), (x1 + w1,
                              y1 + h1), (100, 100, 255), 2)  # Light blue
                cv2.rectangle(output_image, (x2, y2), (x2 + w2,
                              y2 + h2), (100, 100, 255), 2)  # Light blue

                # Calculate the individual distances for visualization
                _, _, dist1 = self.get_distance_to_point(
                    center1[0], center1[1])
                if isinstance(dist1, np.ndarray):
                    dist1 = float(dist1)

                # Publish the distance to the nearest contour
                self.distance_pub.publish(Float32(dist1))

                # Add note about insufficient separation
                # Ensure line_separation is a float
                if isinstance(line_separation, np.ndarray):
                    line_separation = float(line_separation)

                mid_x = (center1[0] + center2[0]) // 2
                mid_y = (center1[1] + center2[1]) // 2
                cv2.putText(output_image, f"Sep: {line_separation:.3f}m (too close)",
                            (mid_x - 100, mid_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Not enough valid contours
            self.blue_double_pub.publish(Float32(float('inf')))
            self.distance_pub.publish(Float32(float('inf')))

            # Still draw any valid contours we found
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h),
                              (100, 100, 255), 2)  # Light blue

        # Publish the visualization image
        image_message = self._bridge.cv2_to_compressed_imgmsg(output_image)
        self.img_pub.publish(image_message)


if __name__ == '__main__':
    node = BlueDetectionNode(node_name='blue_detection_node')
    rospy.spin()
