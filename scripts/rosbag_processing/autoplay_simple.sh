#!/bin/sh
for filename in *.bag; do
    python ~/WRK/ws/ada_ycb_ws/scripts/rosbagToJson_auto.py --infile ${filename}
    mv ${filename} ./processed/${filename}
done
