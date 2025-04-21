#!/usr/bin/env python3

import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Range
from geometry_msgs.msg import Point32
import cv2
from general_navigation import NavigationControl
from std_msgs.msg import ColorRGBA
from duckietown_msgs.msg import LEDPattern
from cv_bridge import CvBridge
from turbojpeg import TurboJPEG
import numpy as np

import dt_apriltags

DEBUG_LANE_FOLLOW = False
DEBUG_TAIL = False


class TailDuckNode(DTROS):
    def __init__(self, node_name):
        super(TailDuckNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']
        self.process_frequency = 5

        # --- Apriltag detection setup ---
        self.detector = dt_apriltags.Detector(families="tag36h11")
        self.tag_size = 0.065
        self.last_tag_id = -1

        # --- General collision prevention setup ---
        self.stop_bot = True
        self.tof_sub = rospy.Subscriber("/" + self.veh + "/front_center_tof_driver_node/range",
                                        Range,
                                        self.cbTOF,
                                        queue_size=1)

        # --- Duckiebot Tailing setup ---
        self.last_stamp = rospy.Time.now()
        self.circlepattern_dims = [7, 3]
        self.blobdetector_min_area = 10
        self.blobdetector_min_dist_between_blobs = 2
        self.cbParametersChanged()
        self.last_seen = rospy.Time(0)
        self.tail_timeout = rospy.Duration(0.65)
        self.last_tail_error = (0, 0)
        self.last_offset = 0
        self.blue_direction = 'left'

        # --- Lane following setup ---
        self.ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
        self.offset = 230
        self.P = 0.025
        self.D = -0.0025
        self.I = 0
        self.last_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()

        # ---- Red intersection setup
        self.stopped_at_red = False
        self.time_of_red_stop = rospy.get_time()
        self.red_cooldown_duration = 10
        self.red_stops_count = 0

        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.image_callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.pub_mask = rospy.Publisher(
            f"/{self.veh}/combined/mask/compressed",
            CompressedImage, queue_size=1
        )
        self.pub_blue_debug = rospy.Publisher(
            f"/{self.veh}/debug/blue_bot_detection/compressed",
            CompressedImage, queue_size=1
        )

        self.pub_circlepattern_image = rospy.Publisher(
            "/{}/duckiebot_detection_node/detection_image/compressed".format(os.environ['VEHICLE_NAME']), CompressedImage, queue_size=1)
        self.log("Detection Initialization completed.")

        self.LEDspattern = LEDPattern()
        self.light_color_list = [  # init led lights
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
        ]
        self.count = 0

        self.pub_leds = rospy.Publisher(
            f"/{self.veh}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        self.nav = NavigationControl()
        self.velocity = 0.28
        self.omega = 0
        self.nav.publish_velocity(0, self.omega)

        rospy.on_shutdown(self.hook)

    def detect_apriltag(self, image_cv):
        """
        Detects apriltags in the image and returns the tag ID and its position.
        """
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=True, camera_params=(
            self.camera_matrix, self.camera_distortion), tag_size=self.tag_size)

        closest = 0
        if tags:
            for tag in tags:
                tag_id = tag.tag_id
                corners = tag.corners
                diff = abs(corners[0]-corners[1])
                if diff > closest:
                    closest = diff
                    tag_id = tag.tag_id
                return int(tag_id)

        return 0

    def stop_at_red(self):
        if not self.stopped_at_red:
            rospy.loginfo("Stopping at red")
            self.time_of_red_stop = rospy.get_time()
            self.nav.stop(3)
            self.stopped_at_red = True

    def publish_LED_pattern(self):
        # Publish the LED pattern to the led_emitter_node
        self.LEDspattern.rgb_vals = []
        for i in range(5):
            rgba = ColorRGBA()
            rgba.r = self.light_color_list[i][0]
            rgba.g = self.light_color_list[i][1]
            rgba.b = self.light_color_list[i][2]
            rgba.a = self.light_color_list[i][3]

            self.LEDspattern.rgb_vals.append(rgba)
        self.pub_leds.publish(self.LEDspattern)

    def set_led_color(self, colors):
        # Set the color of the LEDs

        # colors should be a list of length 5 with
        # each element being a list of length 4
        for i in range(len(self.light_color_list)):
            if len(colors[i]) == 3:
                self.light_color_list[i] = colors[i] + [1]
            else:
                self.light_color_list[i] = colors[i]

        self.publish_LED_pattern()

    def cbTOF(self, msg):
        if 0.05 < msg.range <= 0.2:
            # rospy.loginfo(f"Detected object at : {msg.range}")
            self.stop_bot = True
        else:
            self.stop_bot = False

    def cbParametersChanged(self):
        self.publish_duration = rospy.Duration.from_sec(
            1.0 / self.process_frequency)

        # 1) Improve contrast: CLAHE on the gray channel
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 2) Blob detector tuning
        params = cv2.SimpleBlobDetector_Params()

        # Thresholding: scan from dark to bright
        params.minThreshold = 10
        params.maxThreshold = 200
        params.thresholdStep = 10

        # Keep only roughly circle‑shaped blobs
        params.filterByArea = True
        params.minArea = 20           # increase this if you get too many tiny blobs
        params.maxArea = 5000         # decrease if you pick up big glare patches

        params.filterByCircularity = True
        params.minCircularity = 0.7

        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        params.filterByConvexity = True
        params.minConvexity = 0.8

        # (Optionally) filter by color if your dots are always dark or always light
        # params.filterByColor = True
        # params.blobColor = 0

        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)

    def detect_bot(self, image_cv):
        """
        Runs a oneshot grid detection on a prefiltered image, logs timing, and
        always publishes a debug view showing either the corners+width or "No pattern".
        """
        # --- 1) Pre‑process ---
        #  a) Gaussian blur to smooth noise
        blurred = cv2.GaussianBlur(image_cv, (5, 5), 0)
        #  b) Equalize the V channel for consistent contrast
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        proc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        # --- 2) Detect & time it ---
        t0 = rospy.get_time()
        flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
        found, centers = cv2.findCirclesGrid(
            gray,
            patternSize=tuple(self.circlepattern_dims),
            flags=flags,
            blobDetector=self.simple_blob_detector,
        )
        dt = (rospy.get_time() - t0) * 1000
        # rospy.loginfo(f"CircleGrid dt={dt:.1f}ms, found={found}")

        # Prepare a debug copy
        debug = image_cv.copy()

        if found:
            # compute width + offset
            xs = centers[:, 0, 0]
            pattern_width = float(np.max(xs) - np.min(xs))
            error_distance = 100.0 - pattern_width
            center_offset = float(np.mean(xs) - (image_cv.shape[1] / 2))
            self.last_offset = center_offset

            # annotate
            cv2.drawChessboardCorners(debug, tuple(
                self.circlepattern_dims), centers, found)
            cv2.putText(
                debug, f"W: {pattern_width:.1f}px, last_offser: {self.last_offset}",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2, cv2.LINE_AA
            )

            # update last‑seen for your hold logic
            self.last_seen = rospy.Time.now()
            self.last_pattern = (error_distance, center_offset)

            result = (error_distance, center_offset)

        else:
            # annotate “no pattern”
            cv2.putText(
                debug, "No pattern",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA
            )

            # if you want to hold the last good result up to tail_timeout:
            if (rospy.Time.now() - self.last_seen) < self.tail_timeout:
                result = self.last_pattern
            else:
                result = None
                # rospy.loginfo("Not detected")

        # --- 3) Always publish debug image ---
        # if DEBUG_TAIL:
        imgmsg = self.bridge.cv2_to_compressed_imgmsg(debug)
        self.pub_circlepattern_image.publish(imgmsg)

        return result

    def detect_blue_bot(self, image_cv):
        """
        Detects the blue trailing bot in the image and returns "left"/"right" 
        based on its position. Also publishes a debug image with the contour overlaid.
        """
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([106, 68, 0])
        upper_blue = np.array([151, 255, 145])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = image_cv.copy()

        direction = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                blue_center = x + w // 2

                # draw a box around the detected bot
                cv2.rectangle(debug, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # draw center line
                cv2.line(debug,
                         (blue_center, 0),
                         (blue_center, debug.shape[0]),
                         (255, 0, 0), 1)

                if blue_center < debug.shape[1] // 2:
                    direction = "left"
                    cv2.putText(debug, "LEFT", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    direction = "right"
                    cv2.putText(debug, "RIGHT", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # publish debug image
        if DEBUG_TAIL:
            blue_dbg_msg = self.bridge.cv2_to_compressed_imgmsg(debug)
            self.pub_blue_debug.publish(blue_dbg_msg)

        return direction

    def detect_red_intersection(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_ranges = {'lower': np.array(
            [0, 150, 50]), 'upper': np.array([10, 255, 255])}

        mask = cv2.inRange(hsv, red_ranges['lower'], red_ranges['upper'])
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Estimate distance based on contour position
                image_height = image.shape[0]
                distance = (image_height - (y + h)) / image_height

                return True, distance * 100

        return False, float('inf')

    def lane_detect(self, image_cv):
        # Crops and finds the lane; sets self.proportional
        crop = image_cv[300:, :, :]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        def find_largest(mask):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_area = 20
            max_idx = -1
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_idx = i
            return contours, max_idx

        # Yellow mask first
        yellow_mask = cv2.inRange(
            crop_hsv, self.ROAD_MASK[0], self.ROAD_MASK[1])
        contours, idx = find_largest(yellow_mask)
        following_white = False
        mask_used = yellow_mask

        # Fallback: white lane
        if idx == -1:
            white_lower = np.array([120, 18, 155], np.uint8)
            white_upper = np.array([128, 39, 255], np.uint8)
            white_mask = cv2.inRange(crop_hsv, white_lower, white_upper)
            contours, idx = find_largest(white_mask)
            mask_used = white_mask
            following_white = True if idx != -1 else False

        # Compute centroid & proportional error
        if idx != -1:
            M = cv2.moments(contours[idx])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                offset = -(self.offset +
                           90) if following_white else self.offset
                self.proportional = cx - int(crop.shape[1] / 2) + offset

                # Debug draw
                if DEBUG_LANE_FOLLOW:
                    color = (255, 0, 0) if following_white else (0, 255, 0)
                    cv2.drawContours(crop, contours, idx, color, 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            else:
                self.proportional = None
        else:
            self.proportional = None

        # Publish debug mask
        if DEBUG_LANE_FOLLOW:
            debug_img = cv2.bitwise_and(crop, crop, mask=mask_used)
            msg_out = CompressedImage(
                format="jpeg", data=self.jpeg.encode(debug_img)
            )
            self.pub_mask.publish(msg_out)

    def image_callback(self, msg):
        image_cv = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        now = rospy.Time.now()

        if self.count < 1:
            self.set_led_color(self.light_color_list)
            self.count = 1

        if now - self.last_stamp < self.publish_duration:
            return
        self.last_stamp = now

        # Detect the apriltag in the image
        # tag_id = self.detect_apriltag(image_cv)

        # Always stop at red if not stopped already
        stopline_detected, distance = self.detect_red_intersection(image_cv)

        if stopline_detected and distance < 50:
            blue_direction = self.detect_blue_bot(image_cv)
            if blue_direction is not None:
                self.blue_direction = blue_direction

        # if stopline_detected and distance < 30 and (rospy.get_time() - self.time_of_red_stop) > self.red_cooldown_duration:
        #     self.stopped_at_red = False
        #     self.stop_at_red()

        #     if self.red_stops_count == 0:
        #         if self.blue_direction == "left":
        #             self.nav.turn_left(0.4, 1.5, extra=1.0)
        #             rospy.loginfo("Left turn")
        #         elif self.blue_direction == "right":
        #             self.nav.move_straight(0.35)
        #             self.nav.turn_right(0, -2.5)
        #             rospy.loginfo("Right turn")
        #         self.red_stops_count += 1

        #     elif self.red_stops_count == 1:
        #         self.nav.move_straight(0.5)
        #         rospy.loginfo("Straight")
        #         self.red_stops_count += 1

        #     elif self.red_stops_count == 2:
        #         if self.blue_direction == "left":
        #             self.nav.turn_left(0.4, 1.5, extra=1.0)
        #             rospy.loginfo("Left turn")
        #         elif self.blue_direction == "right":
        #             self.nav.move_straight(0.35)
        #             self.nav.turn_right(0, -2.5)
        #             rospy.loginfo("Right turn")
        #         self.red_stops_count += 1

        #     elif self.red_stops_count == 3:
        #         # logic if a tag was seen
        #         if tag_id is not None:
        #             if tag_id == 48:
        #                 rospy.loginfo("Turning LEFT at AprilTag 21")
        #                 self.nav.turn_left()
        #             elif tag_id == 50:
        #                 rospy.loginfo("Turning RIGHT at AprilTag 59")
        #                 self.nav.turn_right()
        #             else:
        #                 rospy.logwarn(f"Unknown tag ID: {tag_id}")
        #         else:
        #             rospy.loginfo("No tag seen. Proceeding forward.")
        #         self.red_stops_count += 1

        #     elif self.red_stops_count == 4:
        #         # logic if a tag was seen
        #         if tag_id is not None:
        #             if tag_id == 48:
        #                 rospy.loginfo("Turning LEFT at AprilTag 21")
        #                 self.nav.turn_left()
        #             elif tag_id == 50:
        #                 rospy.loginfo("Turning RIGHT at AprilTag 59")
        #                 self.nav.turn_right()
        #             else:
        #                 rospy.logwarn(f"Unknown tag ID: {tag_id}")
        #         else:
        #             rospy.loginfo("No tag seen. Proceeding forward.")
        #         self.red_stops_count += 1

        #     elif self.red_stops_count == 5:
        #         # TODO: Stage 4 Parking
        #         pass

            rospy.loginfo(self.red_stops_count)

        # lane-follow if bot not seen
        if ((now - self.last_seen) >= self.tail_timeout):
            # self.tailing = False
            self.lane_detect(image_cv)

            if self.proportional is None:
                # v = self.velocity
                self.omega = 0
                self.last_error = 0
                self.integral = 0
            else:
                current_time = rospy.get_time()
                dt = current_time - self.last_time
                if dt > 0:
                    d_error = (self.proportional - self.last_error) / dt
                    self.integral += self.proportional * dt
                else:
                    d_error = 0

                Pterm = -self.proportional * self.P
                Dterm = d_error * self.D
                Iterm = self.I * self.integral

                # v = self.velocity
                self.omega = Pterm + Dterm + Iterm

                self.last_error = self.proportional
                self.last_time = current_time
            # rospy.loginfo(f"[Lane Following] v={self.velocity:.2f}, omega={self.omega:.2f}")
            self.nav.publish_velocity(self.velocity, self.omega)

        tail = self.detect_bot(image_cv)
        if tail is not None:
            # self.set_led_color([
            #                     [0, 0, 0, 0],
            #                     [0, 0, 1, 0],
            #                     [0, 0, 0, 0],
            #                     [0, 0, 1, 1],
            #                     [0, 0, 0, 1],
            #                      ])
            error_distance, offset = tail

            # Tuning parameters
            Kp_dist = 0.013
            Kp_angle = -0.005

            # Compute velocity and omega based on error
            v = Kp_dist * error_distance
            omega = Kp_angle * offset

            # Limit speed to avoid overshooting
            v = max(min(v, 0.3), 0.05) if v > 0 else 0

            # Preventing collision is highest priority
            if self.stop_bot:
                self.nav.publish_velocity(0, 0)
                return

            # rospy.loginfo(f"[Tailing] error={error_distance:.1f}, offset={offset:.1f} => v={v:.2f}, omega={omega:.2f}")
            self.nav.publish_velocity(v, omega)
            return

    def hook(self):
        print("SHUTTING DOWN")
        for i in range(8):
            self.nav.publish_velocity(0, 0)


if __name__ == '__main__':
    # create the node
    node = TailDuckNode(node_name='tail_duck_node')
    rospy.spin()


# #!/usr/bin/env python3

# import os
# import numpy as np
# import rospy
# import cv2
# from cv_bridge import CvBridge

# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# from duckietown_msgs.msg import Twist2DStamped, BoolStamped, LEDPattern
# from std_msgs.msg import ColorRGBA, String


# class CircleGridFollowerNode(DTROS):
#     """
#     Circle Grid Follower Node with enhanced debug logging.
#     Provides detailed information about detection and control decisions.
#     """

#     def __init__(self, node_name):
#         super(CircleGridFollowerNode, self).__init__(
#             node_name=node_name, node_type=NodeType.PERCEPTION)

#         # Get vehicle name
#         self.vehicle_name = os.environ.get('VEHICLE_NAME')
#         if self.vehicle_name is None:
#             raise ValueError("Environment variable VEHICLE_NAME is not set")

#         # Parameters for blob detection
#         self.process_frequency = rospy.get_param("~process_frequency", 10.0)
#         self.circlepattern_dims = rospy.get_param(
#             "~circlepattern_dims", [7, 3])
#         self.log(f"Circle pattern dimensions: {self.circlepattern_dims}")

#         # Detection parameters
#         # Time before considering pattern lost
#         self.detection_timeout = rospy.Duration(0.65)
#         self.last_seen = rospy.Time(0)

#         # Blob detector parameters
#         self.blobdetector_min_area = rospy.get_param(
#             "~blobdetector_min_area", 20)
#         self.blobdetector_max_area = rospy.get_param(
#             "~blobdetector_max_area", 2000)
#         self.blobdetector_min_dist_between_blobs = rospy.get_param(
#             "~blobdetector_min_dist_between_blobs", 5)

#         # Controller parameters (PID)
#         # Proportional gain for distance
#         self.Kp_dist = rospy.get_param("~Kp_dist", 0.013)
#         # Proportional gain for angle/offset (increased 10x)
#         self.Kp_angle = rospy.get_param("~Kp_angle", -0.05)
#         # Derivative term (increased 2x)
#         self.Kd = rospy.get_param("~Kd", 0.004)

#         # Add motor deadband compensation
#         # Minimum value to overcome motor deadband
#         self.motor_deadband = rospy.get_param("~motor_deadband", 2.0)
#         self.min_turn_threshold = 30  # Pixel offset threshold for applying minimum turning

#         self.log(
#             f"Control parameters - Kp_dist: {self.Kp_dist}, Kp_angle: {self.Kp_angle}, Kd: {self.Kd}")
#         self.log(f"Motor deadband compensation: {self.motor_deadband}")

#         # Pattern tracking variables
#         self.last_pattern = (0, 0)  # (error_distance, center_offset)
#         self.target_width = 80.0  # Target width for the pattern
#         self.log(f"Target pattern width: {self.target_width} pixels")

#         # Controller state variables
#         self.last_error = 0
#         self.last_time = rospy.get_time()

#         # Debug counters
#         self.frame_count = 0
#         self.detection_count = 0
#         self.missed_count = 0

#         # Initialize timing variables
#         self.last_stamp = rospy.Time.now()
#         self.publish_duration = rospy.Duration.from_sec(
#             1.0 / self.process_frequency)

#         # Detection state
#         self.pattern_detected = False

#         # Safety parameters
#         self.stop_bot = False  # Will be set to True if obstacle detected
#         self.min_velocity = 0.05
#         self.max_velocity = 0.3
#         self.log(
#             f"Velocity limits - min: {self.min_velocity}, max: {self.max_velocity}")

#         # Initialize bridge
#         self.bridge = CvBridge()

#         # Setup blob detector
#         self.setup_blob_detector()

#         # Initialize CLAHE for contrast enhancement
#         self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#         # Publishers
#         # Detection status
#         self.pub_detection = rospy.Publisher(
#             f"/{self.vehicle_name}/circle_grid_follower/detection",
#             BoolStamped,
#             queue_size=1
#         )

#         # Command velocity
#         self.pub_cmd_vel = rospy.Publisher(
#             f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
#             Twist2DStamped,
#             queue_size=1
#         )

#         # Status publisher for debug logging
#         self.pub_status = rospy.Publisher(
#             f"/{self.vehicle_name}/circle_grid_follower/status",
#             String,
#             queue_size=1
#         )

#         # LED pattern publisher (to turn off LEDs)
#         self.led_pattern_pub = rospy.Publisher(
#             f"/{self.vehicle_name}/led_emitter_node/led_pattern",
#             LEDPattern,
#             queue_size=1
#         )

#         # Camera subscriber
#         self.img_sub = rospy.Subscriber(
#             f"/{self.vehicle_name}/camera_node/image/compressed",
#             CompressedImage,
#             self.image_callback,
#             queue_size=1,
#             buff_size="20MB"
#         )

#         # Turn off LEDs to avoid reflection interference
#         self.turn_off_leds()

#         # Set up heartbeat timer to regularly print detection statistics
#         self.stats_timer = rospy.Timer(rospy.Duration(5.0), self.print_stats)

#         self.log("Circle Grid Follower Node initialized with debug logging")
#         rospy.sleep(2)

#     def setup_blob_detector(self):
#         """Configure blob detector for circle grid detection"""
#         params = cv2.SimpleBlobDetector_Params()

#         # Thresholding parameters
#         params.minThreshold = 10
#         params.maxThreshold = 200
#         params.thresholdStep = 10

#         # Shape filter parameters
#         params.filterByArea = True
#         params.minArea = self.blobdetector_min_area
#         params.maxArea = self.blobdetector_max_area

#         params.filterByCircularity = True
#         params.minCircularity = 0.7

#         params.filterByInertia = True
#         params.minInertiaRatio = 0.5

#         params.filterByConvexity = True
#         params.minConvexity = 0.8

#         # Minimum distance between blobs
#         params.minDistBetweenBlobs = self.blobdetector_min_dist_between_blobs

#         # Create detector
#         self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)
#         self.log("Blob detector configured")

#     def turn_off_leds(self):
#         """Turn off LEDs to avoid reflections"""
#         pattern_msg = LEDPattern()
#         pattern_msg.header.stamp = rospy.Time.now()

#         # Create LED pattern with all LEDs off
#         rgb_vals = []
#         for i in range(5):
#             color_rgba = ColorRGBA()
#             color_rgba.r = 0.0
#             color_rgba.g = 0.0
#             color_rgba.b = 0.0
#             color_rgba.a = 1.0
#             rgb_vals.append(color_rgba)

#         pattern_msg.rgb_vals = rgb_vals
#         pattern_msg.frequency = 0.0

#         self.led_pattern_pub.publish(pattern_msg)
#         self.log("LEDs turned off for better detection")

#     def print_stats(self, event=None):
#         """Print detection statistics periodically"""
#         if self.frame_count == 0:
#             return

#         detection_rate = (self.detection_count / self.frame_count) * \
#             100 if self.frame_count > 0 else 0
#         miss_rate = (self.missed_count / self.frame_count) * \
#             100 if self.frame_count > 0 else 0

#         self.log(f"STATS: Frames: {self.frame_count}, Detections: {self.detection_count} ({detection_rate:.1f}%), " +
#                  f"Misses: {self.missed_count} ({miss_rate:.1f}%)")

#     def image_callback(self, msg):
#         """Process incoming camera images to detect and follow circle grid"""
#         # Turn off LEDs to avoid reflection interference
#         self.turn_off_leds()
#         self.frame_count += 1
#         now = rospy.Time.now()

#         # Rate limiting
#         if now - self.last_stamp < self.publish_duration:
#             return

#         self.last_stamp = now

#         # Convert compressed image to OpenCV format
#         try:
#             np_arr = np.frombuffer(msg.data, np.uint8)
#             image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         except Exception as e:
#             self.log(f"Error converting image: {e}", 'error')
#             return

#         # Detect the circle grid
#         detection_result = self.detect_circle_grid(image_cv)

#         # Process detection result
#         if detection_result is not None:
#             # Pattern detected
#             error_distance, center_offset = detection_result
#             self.pattern_detected = True
#             self.last_seen = now
#             self.detection_count += 1

#             # Log detection details
#             direction = "RIGHT" if center_offset < 0 else "LEFT"
#             distance_status = "TOO FAR" if error_distance > 0 else "TOO CLOSE"
#             self.log(f"DETECTED: Pattern offset: {center_offset:.1f}px ({direction}), " +
#                      f"Width error: {error_distance:.1f}px ({distance_status})")

#             # Publish detection status
#             self.publish_detection_status(True)

#             # Use PID controller to follow the pattern
#             self.follow_pattern(error_distance, center_offset)

#             # Publish detailed status
#             self.pub_status.publish(
#                 f"DET: dist={error_distance:.1f}, offset={center_offset:.1f}, dir={direction}")
#         else:
#             # Check if we're still within timeout
#             if (now - self.last_seen) < self.detection_timeout:
#                 # Use last known pattern
#                 error_distance, center_offset = self.last_pattern
#                 self.pattern_detected = True

#                 # Log using previous detection
#                 direction = "RIGHT" if center_offset < 0 else "LEFT"
#                 self.log(f"USING LAST: Pattern offset: {center_offset:.1f}px ({direction}), " +
#                          f"Width error: {error_distance:.1f}px, Time since last: {(now-self.last_seen).to_sec():.2f}s")

#                 # Publish detection status
#                 self.publish_detection_status(True)

#                 # Follow using last known position
#                 self.follow_pattern(error_distance, center_offset)

#                 # Publish detailed status
#                 self.pub_status.publish(
#                     f"LAST: dist={error_distance:.1f}, offset={center_offset:.1f}, dir={direction}")
#             else:
#                 # Pattern lost
#                 self.pattern_detected = False
#                 self.missed_count += 1

#                 # Log pattern loss
#                 time_since = (now - self.last_seen).to_sec()
#                 self.log(
#                     f"LOST PATTERN: {time_since:.2f}s since last detection, STOPPING")

#                 # Publish detection status
#                 self.publish_detection_status(False)

#                 # Stop the robot
#                 self.stop()

#                 # Publish detailed status
#                 self.pub_status.publish(
#                     f"LOST: {time_since:.1f}s since last detection")

#     def detect_circle_grid(self, image_cv):
#         """
#         Detect circle grid pattern in the image

#         Args:
#             image_cv: OpenCV image

#         Returns:
#             tuple (error_distance, center_offset) or None if not detected
#         """
#         # Pre-process the image
#         # 1) Convert to grayscale
#         gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

#         # 2) Apply CLAHE for better contrast
#         gray = self.clahe.apply(gray)

#         # 3) Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Find circle grid
#         flags = cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
#         found, centers = cv2.findCirclesGrid(
#             blurred,
#             patternSize=tuple(self.circlepattern_dims),
#             flags=flags,
#             blobDetector=self.simple_blob_detector
#         )

#         if found:
#             # Calculate pattern width and offset
#             xs = centers[:, 0, 0]
#             pattern_width = float(np.max(xs) - np.min(xs))
#             error_distance = self.target_width - pattern_width  # Positive means too far
#             # Positive means pattern is to the right
#             center_offset = float(np.mean(xs) - (image_cv.shape[1] / 2))

#             # Update last known pattern
#             self.last_pattern = (error_distance, center_offset)

#             return (error_distance, center_offset)
#         else:
#             return None

#     def follow_pattern(self, error_distance, center_offset):
#         """
#         PID controller to follow the detected pattern

#         Args:
#             error_distance: Distance error (target - actual width)
#             center_offset: Offset from center of image
#         """
#         current_time = rospy.get_time()
#         dt = current_time - self.last_time
#         if dt <= 0:
#             dt = 0.1  # Avoid division by zero

#         # Compute velocity and omega based on error
#         v = self.Kp_dist * error_distance
#         omega = self.Kp_angle * center_offset

#         # Calculate derivative term for smoother control
#         d_offset = (center_offset - self.last_error) / dt
#         omega += self.Kd * d_offset

#         # Apply aggressive motor deadband compensation
#         # This ensures the turning command is strong enough to overcome motor limitations
#         if abs(center_offset) > self.min_turn_threshold:  # If offset is significant
#             # Apply a deadband offset in the right direction
#             if center_offset > 0:  # Need to turn left
#                 omega = omega + self.motor_deadband
#             else:  # Need to turn right
#                 omega = omega - self.motor_deadband

#             # Ensure minimum turning rate regardless of calculated value
#             if center_offset > 0 and omega < 2.0:  # If turning left but not strongly enough
#                 omega = 2.0
#             elif center_offset < 0 and omega > -2.0:  # If turning right but not strongly enough
#                 omega = -2.0

#             # Temporarily reduce forward speed when turning sharply
#             if abs(omega) > 3.0:
#                 v = max(v * 0.7, 0.15)  # Reduce speed to 70% but keep minimum

#             self.log(
#                 f"TURNING BOOSTED: center_offset={center_offset:.1f}, omega={omega:.3f}")

#         # Log raw control values
#         turn_direction = "LEFT" if omega > 0 else "RIGHT"
#         forward_direction = "FORWARD" if v > 0 else "BACKWARD"
#         self.log(
#             f"CONTROL: Turn {turn_direction} (omega={omega:.3f}), Move {forward_direction} (v={v:.3f})")

#         # Limit speed to avoid overshooting
#         orig_v = v
#         v = min(max(v, self.min_velocity), self.max_velocity) if v > 0 else 0

#         # Limit omega to reasonable values - allow higher maximum for better turning
#         orig_omega = omega
#         max_omega = 10.0  # Increased from 8.0
#         omega = max(min(omega, max_omega), -max_omega)

#         # Log if limits were applied
#         if orig_v != v or orig_omega != omega:
#             self.log(
#                 f"LIMITED: v: {orig_v:.3f} -> {v:.3f}, omega: {orig_omega:.3f} -> {omega:.3f}")

#         # Create velocity command
#         cmd = Twist2DStamped()
#         cmd.header.stamp = rospy.Time.now()
#         cmd.v = v
#         cmd.omega = omega

#         # Publish command
#         self.pub_cmd_vel.publish(cmd)

#         # Update control state
#         self.last_error = center_offset
#         self.last_time = current_time

#     def publish_detection_status(self, detected):
#         """Publish detection status message"""
#         detection_msg = BoolStamped()
#         detection_msg.header.stamp = rospy.Time.now()
#         detection_msg.data = detected
#         self.pub_detection.publish(detection_msg)

#     def stop(self):
#         """Stop the robot"""
#         cmd = Twist2DStamped()
#         cmd.header.stamp = rospy.Time.now()
#         cmd.v = 0
#         cmd.omega = 0
#         self.pub_cmd_vel.publish(cmd)
#         self.log("STOP: Robot stopped due to lost pattern")

#     def on_shutdown(self):
#         """Clean up when node is shut down"""
#         # Stop the robot
#         for i in range(20):  # Send multiple stop commands to ensure it's received
#             self.stop()
#             # Turn off LEDs to avoid reflection interference
#             self.turn_off_leds()
#         self.log("Circle Grid Follower Node shutting down")
#         # Print final statistics
#         self.print_stats()


# if __name__ == '__main__':
#     # Create the node
#     node = CircleGridFollowerNode(node_name='circle_grid_follower_node')
#     # Register shutdown hook
#     rospy.on_shutdown(node.on_shutdown)
#     # Keep the node running
#     rospy.spin()
