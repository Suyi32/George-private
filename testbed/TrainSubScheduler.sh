#!/bin/sh

#./TrainSubScheduler.sh
pathname="./UnifiedModel_PCPO_PPO_ray_ilp.py"


do
python3 $pathname --start_sample $(($VARIABLE*20))
done
