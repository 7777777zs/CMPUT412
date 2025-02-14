#!/usr/bin/env python3

import os
import rospy
import rosbag
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, LEDPattern
from std_msgs.msg import String, ColorRGBA
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion

class RobotState:
    STOP = "STOP"
    TRACING_D = "TRACING_D"
    RETURN = "RETURN"

class LEDColors:
    # Define LED colors as strings
    BLUE = 'blue'
    GREEN = 'green'
    RED = 'red'
    OFF = 'off'

class DShapeNode(DTROS):
    def __init__(self, node_name):
        super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # Get vehicle name
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Publishers
        self.wheels_publisher = rospy.Publisher(
            f'/{self.vehicle_name}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=1
        )
        self.led_publisher = rospy.Publisher(
            f'/{self.vehicle_name}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=1
        )

        # Initialize state variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = None
        self.wheel_distance = 0.1  # Distance between wheels in meters
        self.start_time = None
        
        # Create rosbag
        self.bag = rosbag.Bag(f'odometry_{self.vehicle_name}.bag', 'w')

        # Movement parameters
        self.linear_speed = 0.3      # meters per second
        self.angular_speed = 1.0     # radians per second
        self.straight_distance = 0.3  # meters
        self.radius = 0.3            # meters for semicircle

    def set_led_color(self, color):
        """Set LED color based on state"""
        msg = LEDPattern()
        msg.color_list = [color] * 5  # Set all 5 LEDs to the same color
        msg.frequency = 0.0
        msg.frequency_mask = [0, 0, 0, 0, 0]
        self.led_publisher.publish(msg)

    def update_odometry(self, v_left, v_right, current_time):
        """Update odometry based on wheel velocities"""
        if self.last_time is None:
            self.last_time = current_time
            return
        
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time
        
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / self.wheel_distance
        
        if abs(omega) < 1e-6:
            self.x += v * dt * math.cos(self.theta)
            self.y += v * dt * math.sin(self.theta)
        else:
            self.theta += omega * dt
            self.x += (v/omega) * (math.sin(self.theta + omega * dt) - math.sin(self.theta))
            self.y -= (v/omega) * (math.cos(self.theta + omega * dt) - math.cos(self.theta))
            self.theta = self.theta + omega * dt
        
        # Create and save odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = f"{self.vehicle_name}/odom"
        odom_msg.child_frame_id = f"{self.vehicle_name}/base_link"
        odom_msg.pose.pose.position = Point(x=self.x, y=self.y, z=0.0)
        odom_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, 
                                                  z=math.sin(self.theta/2.0), 
                                                  w=math.cos(self.theta/2.0))
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = omega
        
        self.bag.write(f'/{self.vehicle_name}/odometry', odom_msg, current_time)

    def drive_straight(self, distance, speed, rate):
        """Drive straight for a given distance"""
        duration = abs(distance / speed)
        start_time = rospy.Time.now()
        
        while (rospy.Time.now() - start_time) < rospy.Duration(duration) and not rospy.is_shutdown():
            cmd = WheelsCmdStamped(vel_left=speed, vel_right=speed)
            self.wheels_publisher.publish(cmd)
            self.update_odometry(speed, speed, rospy.Time.now())
            rate.sleep()
        
        # Stop the robot
        cmd = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        self.wheels_publisher.publish(cmd)
        rate.sleep()

    def trace_semicircle(self, radius, rate):
        """Trace a semicircle"""
        # Calculate the angular velocity needed for the semicircle
        v = self.linear_speed  # Linear velocity
        omega = v / radius     # Angular velocity for circular motion
        
        # Calculate wheel velocities for circular motion
        v_right = v * (1 + self.wheel_distance / (2 * radius))  # Outer wheel
        v_left = v * (1 - self.wheel_distance / (2 * radius))   # Inner wheel
        
        # Calculate duration for half circle (pi radians)
        duration = (math.pi * radius) / v
        
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time) < rospy.Duration(duration) and not rospy.is_shutdown():
            cmd = WheelsCmdStamped(vel_left=v_left, vel_right=v_right)
            self.wheels_publisher.publish(cmd)
            self.update_odometry(v_left, v_right, rospy.Time.now())
            rate.sleep()
        
        # Stop the robot
        cmd = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        self.wheels_publisher.publish(cmd)
        rate.sleep()

    def run(self):
        rate = rospy.Rate(10)
        self.start_time = rospy.Time.now()

        try:
            # State 1: Initial stop with blue LED
            self.set_led_color(LEDColors.BLUE)
            rospy.sleep(5.0)  # Wait for 5 seconds

            # State 2: Tracing D shape with green LED
            self.set_led_color(LEDColors.GREEN)
            
            # Straight line
            self.drive_straight(self.straight_distance, self.linear_speed, rate)
            rospy.sleep(1.0)  # Small pause between movements
            
            # Semicircle
            self.trace_semicircle(self.radius, rate)
            rospy.sleep(1.0)  # Small pause between movements

            # State 3: Return to start with red LED
            self.set_led_color(LEDColors.RED)
            
            # Return path
            self.drive_straight(-self.straight_distance, self.linear_speed, rate)
            
            # Final stop with blue LED
            self.set_led_color(LEDColors.BLUE)
            rospy.sleep(5.0)  # Wait for 5 seconds

            # Calculate total execution time
            total_time = (rospy.Time.now() - self.start_time).to_sec()
            self.loginfo(f"Total execution time: {total_time:.2f} seconds")

        except Exception as e:
            self.loginfo(f"An error occurred: {str(e)}")
            raise

    def on_shutdown(self):
        """Cleanup on shutdown"""
        try:
            # Stop the robot
            stop_cmd = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
            self.wheels_publisher.publish(stop_cmd)
            
            # Turn off LEDs
            self.set_led_color(LEDColors.OFF)
            
            # Close the bag file
            if hasattr(self, 'bag') and self.bag is not None:
                self.bag.close()
            
            self.loginfo("Shutting down: Wheels stopped, LEDs off, and bag file saved.")
        except Exception as e:
            self.loginfo(f"Error during shutdown: {str(e)}")

if __name__ == '__main__':
    node = DShapeNode(node_name='d_shape_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An error occurred: {str(e)}")
    finally:
        if hasattr(node, 'bag') and node.bag is not None:
            node.bag.close()
        rospy.signal_shutdown("Node completed")



