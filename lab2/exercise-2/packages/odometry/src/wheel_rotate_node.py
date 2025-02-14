#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
import math

# Configuration for driving distances and speeds.
# Assuming that a wheel command of 0.5 corresponds to ~0.5 m/s.
FORWARD_SPEED = 0.5      # 0.5 m/s forward (not used for rotation)
BACKWARD_SPEED = 0.5     # 0.5 m/s backward (not used for rotation)
DISTANCE = 1.25          # 1.25 meters (not used for rotation)

# Configuration for rotation
ROTATIONAL_SPEED = 0.2   # Adjust this value (percentage of max speed)
                        # until the rotation is close to accurate
CLOCKWISE = -1          # Right wheel forward, left wheel backward
COUNTERCLOCKWISE = 1    # Left wheel forward, right wheel backward

class WheelControlNode(DTROS):
    def __init__(self, node_name):
        # initialize DTROS parent class
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # Get the vehicle name and build the wheels topic
        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

    def rotate(self, angle_radians, direction):
        """
        Rotates the Duckiebot by a specified angle.

        Args:
            angle_radians (float): The angle to rotate in radians.
            direction (int): 1 for counterclockwise, -1 for clockwise.
        """
        angular_speed = ROTATIONAL_SPEED  # radians per second (adjust this!)

        # Calculate the time required for rotation
        # angle = angular_speed * time => time = angle / angular_speed
        rotation_duration = abs(angle_radians / angular_speed)

        # Set wheel velocities based on direction
        vel_left = direction * ROTATIONAL_SPEED
        vel_right = -direction * ROTATIONAL_SPEED

        # Create the wheel command message
        rotate_cmd = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)

        # Publish the command for the calculated duration
        rate = rospy.Rate(10)  # 10 Hz
        start_time = rospy.Time.now()

        self.loginfo(f"Rotating for {rotation_duration:.2f} seconds")
        while (rospy.Time.now() - start_time) < rospy.Duration(rotation_duration) and not rospy.is_shutdown():
            self._publisher.publish(rotate_cmd)
            rate.sleep()

        # Stop the wheels after rotation
        self.stop()
        rospy.sleep(0.1)  # Short pause after stopping

    def stop(self):
        """Stops the Duckiebot."""
        stop_cmd = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        self._publisher.publish(stop_cmd)

    def run(self):
        # Rotate 90 degrees (pi/2 radians) clockwise
        self.rotate(math.pi/2, CLOCKWISE)
        rospy.loginfo("Finished rotating clockwise.")

        # Rotate back to 0 degrees (pi/2 radians) counterclockwise
        self.rotate(math.pi/2, COUNTERCLOCKWISE)
        rospy.loginfo("Finished rotating counterclockwise.")

        rospy.signal_shutdown("Finished rotation sequence.")

    def on_shutdown(self):
        # Safely stop the robot on shutdown.
        self.stop()
        self.loginfo("Shutting down: Wheels stopped.")

if __name__ == '__main__':
    # Initialize the node.
    node = WheelControlNode(node_name='wheel_control_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.spin()



