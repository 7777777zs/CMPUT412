#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch the camera distortion node
rosrun computer_vision lane_following_node.py

# wait for app to end
dt-launchfile-join
