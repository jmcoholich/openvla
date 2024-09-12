#!/bin/bash

# Send commands to Terminator to split and run your desired commands
xdotool key ctrl+shift+e
sleep 0.25
xdotool type 'ssh 172.16.0.3'
sleep 0.25
xdotool key Return
sleep 2.0
xdotool type 'cd deoxys_control/deoxys && ./auto_scripts/auto_arm.sh config/charmander.yml'
sleep 0.25
xdotool key Return
sleep 0.25

xdotool key ctrl+shift+o
sleep 0.25
xdotool type 'ssh 172.16.0.3'
sleep 0.25
xdotool key Return
sleep 2.0
xdotool type 'cd deoxys_control/deoxys && ./auto_scripts/auto_gripper.sh config/charmander.yml'
sleep 0.25
xdotool key Return
sleep 0.25

xdotool key ctrl+shift+o
sleep 0.25
xdotool type 'conda activate openteach && cd ~/openteach'
sleep 0.25
xdotool key Return
sleep 0.25
xdotool type 'python robot_camera.py'
sleep 0.25
xdotool key Return



exit 0
