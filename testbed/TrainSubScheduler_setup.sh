#!/bin/sh
pathname="./UnifiedModel_PCPO_PPO_ray_ilp.py"

screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -dmS MySessionName3 &
screen -dmS MySessionName4 &
screen -dmS MySessionName5 &
screen -dmS MySessionName6 &
screen -dmS MySessionName7 &
screen -dmS MySessionName8 &
screen -dmS MySessionName9 &
screen -dmS MySessionName10 &
#screen -dmS MySessionName11 &
#screen -dmS MySessionName12 &
#screen -dmS MySessionName13 &
#screen -dmS MySessionName14 &
screen -S MySessionName0 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 0
" &
screen -S MySessionName1 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 10
"&
screen -S MySessionName2 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 20
"&
screen -S MySessionName3 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 30
" &
screen -S MySessionName4 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 40
"&
screen -S MySessionName5 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 50
"&
screen -S MySessionName6 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 60
"&
screen -S MySessionName7 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 70
"&
screen -S MySessionName8 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 80
"&
screen -S MySessionName9 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 90
"&
screen -S MySessionName10 -p 0 -X stuff "/workspace/George-private/testbed;  python3 $pathname --start_sample 100
"