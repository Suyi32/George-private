#!/bin/sh

#./TrainSubScheduler.sh
pathname="./UnifiedModel_PCPO_PPO_ray_ilp.py"

for VARIABLE in 0 1 2 3 4 5 6 7 8 9
do
python3 $pathname --start_sample $(($VARIABLE*20))
done
