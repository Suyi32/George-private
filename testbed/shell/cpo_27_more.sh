#!/bin/sh

pathname="CPO_27_more.py"

for VARIABLE in 0 1 2 3 4 5 6 7 8 9
do
#python3 $pathname --batch_choice $(($VARIABLE + $1))
python3 -u -W ignore PPPOWithoutSubScheduler_PCPO_PPO.py --batch_choice $(($VARIABLE + $1)) --clip_eps 0.1 --safety_requirement 0.02 --lr 0.001
done

# etc.