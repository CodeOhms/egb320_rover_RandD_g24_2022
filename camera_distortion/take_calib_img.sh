#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")
echo "raspistill -t 0 -k -n -vf -hf -e png -o calib_imgs/$1_$DATE_%04d.png"
raspistill -t 0 -k -n -vf -hf -e png -o calib_imgs/$1_$DATE_%04d.png
