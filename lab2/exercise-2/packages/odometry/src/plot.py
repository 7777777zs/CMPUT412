#!/usr/bin/env python3

import os
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_trajectory():
    # Get vehicle name from environment variable
    vehicle_name = os.environ['VEHICLE_NAME']
    
    # Initialize arrays to store trajectory
    x_points = [0.0]
    y_points = [0.0]
    
    # Open the rosbag file with the same name pattern used in recording
    bag_filename = f'odometry_{vehicle_name}.bag'
    
    try:
        # Open the bag file
        bag = rosbag.Bag(bag_filename)
        
        # Read messages from the odometry topic
        for topic, msg, t in bag.read_messages(topics=[f'/{vehicle_name}/odometry']):
            # Extract position directly from odometry message
            x_points.append(msg.pose.pose.position.x)
            y_points.append(msg.pose.pose.position.y)
        
        bag.close()
        
        # Create the plot
        plt.figure(figsize=(10, 10))
        plt.plot(x_points, y_points, 'b-', label='Robot Trajectory')
        plt.plot(x_points[0], y_points[0], 'go', label='Start')
        plt.plot(x_points[-1], y_points[-1], 'ro', label='End')
        
        plt.title(f'Duckiebot {vehicle_name} Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        # Save the plot
        plt.savefig(f'trajectory_{vehicle_name}.png')
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find bag file '{bag_filename}'")
    except Exception as e:
        print(f"Error while processing bag file: {e}")
    finally:
        try:
            bag.close()
        except:
            pass

if __name__ == '__main__':
    plot_trajectory()



