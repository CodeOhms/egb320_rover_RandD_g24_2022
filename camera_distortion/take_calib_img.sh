#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")
echo "raspistill -vf -hf -t 1000 -e png -o calib_imgs/$1_$DATE.png"
raspistill -vf -hf -t 1000 -e png -o calib_imgs/$1_$DATE.png
