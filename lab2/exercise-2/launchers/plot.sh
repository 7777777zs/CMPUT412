#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun odometry plot.py

# wait for app to end
dt-launchfile-join
