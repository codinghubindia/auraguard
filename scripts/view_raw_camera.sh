#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /home/maxx/naviguard_ws/install/setup.bash
python3 /home/maxx/naviguard_ws/scripts/image_viewer.py /camera/image_raw "$@"
