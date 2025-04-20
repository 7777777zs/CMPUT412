#!/usr/bin/env python3

import os
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
from std_msgs.msg import Float32, String


class BotFollowingNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(BotFollowingNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Get vehicle name
        self.vehicle_name = os.environ.get('VEHICLE_NAME')
        if self.vehicle_name is None:
            raise ValueError("Environment variable VEHICLE_NAME is not set")

        # Initialize controller parameters (using the same tuned values from lane following)
        self.kp = 0.05  # Proportional gain - increased for more responsive turning
        self.kd = 0.02  # Derivative gain - increased for better damping
        self.ki = 0.001  # Integral gain

        # Controller state
        self.error_x = 0
        self.last_error_x = 0
        self.integral_x = 0
        self.derivative_x = 0

        # Speed parameters
        self.normal_speed = 0.4      # Regular forward speed
        self.slow_speed = 0.2        # Slower speed when far from center
        self.current_speed = self.normal_speed

        # Turning parameters
        self.min_omega = 0.5  # Minimum angular velocity to ensure turning

        # Last update time
        self.last_time = rospy.get_time()

        # Board detection status
        self.board_detected = False
        self.last_board_detected = False
        self.last_detection_time = rospy.get_time()
        self.detection_timeout = 1.0  # Seconds before considering board lost

        # Missed detection counter
        self.missed_detections = 0
        self.max_missed_detections = 4  # Stop after 4 consecutive missed detections

        # Board position
        self.board_center_x = 0.0
        self.board_center_y = 0.0
        self.last_valid_center_x = 0.0
        self.last_valid_center_y = 0.0

        # Subscribe to board position topics
        self.center_x_sub = rospy.Subscriber(
            f'/{self.vehicle_name}/blob_detection_node/board_center_x',
            Float32,
            self.center_x_callback
        )

        self.center_y_sub = rospy.Subscriber(
            f'/{self.vehicle_name}/blob_detection_node/board_center_y',
            Float32,
            self.center_y_callback
        )

        self.detection_sub = rospy.Subscriber(
            f'/{self.vehicle_name}/blob_detection_node/detection',
            BoolStamped,
            self.detection_callback
        )

        # Publishers
        # 1. Command velocity publisher
        self.cmd_vel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        # 2. Controller status publisher
        self.status_pub = rospy.Publisher(
            f"/{self.vehicle_name}/checkerboard_following/status",
            String,
            queue_size=1
        )

        # Timer for controller update (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        self.log("Checkerboard following node initialized with PID control")

    def center_x_callback(self, msg):
        """Callback for receiving board's X position"""
        self.board_center_x = msg.data
        if self.board_detected:
            self.last_valid_center_x = self.board_center_x

    def center_y_callback(self, msg):
        """Callback for receiving board's Y position"""
        self.board_center_y = msg.data
        if self.board_detected:
            self.last_valid_center_y = self.board_center_y

    def detection_callback(self, msg):
        """Callback for receiving board detection status"""
        self.last_board_detected = self.board_detected
        self.board_detected = msg.data

        # Check for detection status change
        if not self.last_board_detected and self.board_detected:
            # Board was just detected after being lost
            self.missed_detections = 0
            self.log("Checkerboard detected")
        elif self.last_board_detected and not self.board_detected:
            # Board was just lost
            self.missed_detections += 1
            self.log(
                f"Checkerboard lost (missed: {self.missed_detections}/{self.max_missed_detections})")

        if self.board_detected:
            self.last_detection_time = rospy.get_time()

    def is_board_visible(self):
        """Check if board is currently visible or if we're within allowed missed detections"""
        if self.board_detected:
            return True
        elif self.missed_detections < self.max_missed_detections:
            # Use previous position if within allowed missed detections
            return True
        else:
            return False

    def control_loop(self, event):
        """PID control loop for checkerboard following"""
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        if dt == 0:
            return
        self.last_time = current_time

        # If board is not visible and we've exceeded missed detection limit, stop
        if not self.is_board_visible():
            self.stop()
            self.status_pub.publish(
                "Board lost, robot stopped (too many missed detections)")
            return

        # Use current position if detected, otherwise use last valid position
        center_x = self.board_center_x if self.board_detected else self.last_valid_center_x
        center_y = self.board_center_y if self.board_detected else self.last_valid_center_y

        # Use x position error (negative because positive x error means we need to turn left)
        self.error_x = -center_x

        # Calculate derivative
        self.derivative_x = (self.error_x - self.last_error_x) / dt

        # Calculate integral with anti-windup
        self.integral_x += self.error_x * dt

        # Anti-windup: Limit integral term
        if self.integral_x > 100:
            self.integral_x = 100
        elif self.integral_x < -100:
            self.integral_x = -100

        # PID controller for omega (rotation)
        omega = self.kp * self.error_x + self.kd * \
            self.derivative_x + self.ki * self.integral_x

        # Ensure minimum turning speed to improve responsiveness
        if abs(self.error_x) > 0.1:  # If error is significant
            if omega > 0:
                omega = max(omega, self.min_omega)
            elif omega < 0:
                omega = min(omega, -self.min_omega)

        # Limit the maximum omega to prevent excessive rotation
        max_omega = 8.0
        omega = max(min(omega, max_omega), -max_omega)

        # Adjust speed based on error magnitude - slow down when far from center
        error_magnitude = abs(self.error_x)
        if error_magnitude > 0.5:  # If more than half way to the edge
            self.current_speed = self.slow_speed
        else:
            # Gradually increase speed as we get closer to center
            self.current_speed = self.slow_speed + \
                (self.normal_speed - self.slow_speed) * (1 - error_magnitude/0.5)

        # Adjust forward/backward motion based on board's Y position
        # Positive y means board is below center (move forward), negative means above (move backward)
        v = self.current_speed
        if center_y > 0.1:  # Board is below center - move forward
            v = self.current_speed
        elif center_y < -0.1:  # Board is above center - move backward
            v = -self.current_speed
        else:  # Board is at good Y position - maintain position
            v = 0

        # Create and publish velocity command
        cmd = Twist2DStamped()
        cmd.v = v
        cmd.omega = omega
        self.cmd_vel_pub.publish(cmd)

        # Save error for next iteration
        self.last_error_x = self.error_x

        # Publish status for debugging/monitoring
        status_msg = (f"Error X: {self.error_x:.2f}, Error Y: {center_y:.2f}, "
                      f"D: {self.derivative_x:.2f}, I: {self.integral_x:.2f}, "
                      f"Omega: {omega:.2f}, Speed: {v:.2f}, Missed: {self.missed_detections}")
        self.status_pub.publish(status_msg)

    def stop(self):
        """Stop the robot by publishing zero velocity"""
        cmd = Twist2DStamped()
        cmd.v = 0
        cmd.omega = 0
        self.cmd_vel_pub.publish(cmd)

    def on_shutdown(self):
        """Clean up when node is shut down"""
        for i in range(5):
            self.stop()
        self.log("Checkerboard following node shutting down, robot stopped")


if __name__ == '__main__':
    # Initialize the node
    checkerboard_following_node = BotFollowingNode(
        node_name='checkerboard_following_node')

    # Register shutdown hook
    rospy.on_shutdown(checkerboard_following_node.on_shutdown)
    rospy.spin()
