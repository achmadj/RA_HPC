#!/bin/bash

# Run the first command in the background
nohup python /clusterfs/students/achmadjae/RA/02_deep3D/script/3d_sho-dwig-iw.py iw > /clusterfs/students/achmadjae/RA/02_deep3D/script/data_log/iw.log &
# Capture its process ID
PID1=$!

# Wait for the first command to finish
wait $PID1

# Run the second command in the background
nohup python /clusterfs/students/achmadjae/RA/02_deep3D/script/main.py iw train > /clusterfs/students/achmadjae/RA/02_deep3D/script/train_log/iw.log &
# Capture its process ID
PID2=$!

# Wait for the second command to finish
wait $PID2

# Run the third command in the background
nohup python /clusterfs/students/achmadjae/RA/02_deep3D/script/main.py iw test > /clusterfs/students/achmadjae/RA/02_deep3D/script/test_log/iw_test.log &
# Capture its process ID
PID3=$!

# Wait for the third command to finish
wait $PID3