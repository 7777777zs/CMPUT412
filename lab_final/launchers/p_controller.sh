#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch the camera distortion node
rosrun computer_vision p_controller_node.py

# wait for app to end
dt-launchfile-join
