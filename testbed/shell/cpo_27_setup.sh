#!/bin/sh

ps -ef | grep python

pkill screen
pkill python
pkill pyhon3
cd /workspace/George-private/testbed
#git pull




screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -S MySessionName0 -p 0 -X stuff "cd /workspace/George-private/testbed;
   sh shell/cpo_27_more.sh 0
" &
screen -S MySessionName1 -p 0 -X stuff "cd /workspace/George-private/testbed;  sh shell/cpo_27_more.sh 10
" &
screen -S MySessionName2 -p 0 -X stuff "cd /workspace/George-private/testbed;  sh shell/cpo_27_more.sh 20
"
screen -r MySessionName0

