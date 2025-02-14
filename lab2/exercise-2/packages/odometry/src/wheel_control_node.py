#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

# Configuration for driving distances and speeds.
# Assuming that a wheel command of 0.5 corresponds to ~0.5 m/s.
FORWARD_SPEED = 0.5      # 0.5 m/s forward
BACKWARD_SPEED = 0.5     # 0.5 m/s backward (magnitude)
DISTANCE = 1.25          # 1.25 meters

class WheelControlNode(DTROS):
    def __init__(self, node_name):
        # initialize DTROS parent class
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # Get the vehicle name and build the wheels topic
        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

    def run(self):
        # Calculate time duration based on distance and speed:
        # time = distance / speed
        forward_duration = DISTANCE / FORWARD_SPEED       # seconds to drive forward
        backward_duration = DISTANCE / BACKWARD_SPEED     # seconds to drive backward

        # Define wheel commands for forward and backward motion.
        forward_cmd = WheelsCmdStamped(vel_left=FORWARD_SPEED, vel_right=FORWARD_SPEED)
        backward_cmd = WheelsCmdStamped(vel_left=-BACKWARD_SPEED, vel_right=-BACKWARD_SPEED)

        # We'll publish commands at 10Hz
        rate = rospy.Rate(10)

        self.loginfo(f"Driving forward for {forward_duration:.2f} seconds")
        start_time = rospy.Time.now()
        # Publish forward command for the required duration.
        while (rospy.Time.now() - start_time) < rospy.Duration(forward_duration) and not rospy.is_shutdown():
            self._publisher.publish(forward_cmd)
            rate.sleep()

        self.loginfo(f"Driving backward for {backward_duration:.2f} seconds")
        start_time = rospy.Time.now()
        # Publish backward command for the required duration.
        while (rospy.Time.now() - start_time) < rospy.Duration(backward_duration) and not rospy.is_shutdown():
            self._publisher.publish(backward_cmd)
            rate.sleep()

        self.loginfo("Stopping wheels")
        # After completing the maneuvers, ensure the wheels are stopped.
        stop_cmd = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        # Publish the stop command a few times to ensure it is received.
        for _ in range(10):
            self._publisher.publish(stop_cmd)
            rate.sleep()

    def on_shutdown(self):
        # Safely stop the robot on shutdown.
        stop_cmd = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        self._publisher.publish(stop_cmd)
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