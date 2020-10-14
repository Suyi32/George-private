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
screen -dmS MySessionName4 &
screen -dmS MySessionName5 &
screen -dmS MySessionName3 &
screen -S MySessionName0 -p 0 -X stuff "cd /workspace/George-private/testbed;
sh shell/cpo_27_more.sh 0
" &
screen -S MySessionName1 -p 0 -X stuff "cd /workspace/George-private/testbed;
  sh shell/cpo_27_more.sh 5
" &
screen -S MySessionName2 -p 0 -X stuff "cd /workspace/George-private/testbed;
  sh shell/cpo_27_more.sh 10
" &
screen -S MySessionName3 -p 0 -X stuff "cd /workspace/George-private/testbed;
sh shell/cpo_27_more.sh 15
" &
screen -S MySessionName4 -p 0 -X stuff "cd /workspace/George-private/testbed;
  sh shell/cpo_27_more.sh 20
" &
screen -S MySessionName5 -p 0 -X stuff "cd /workspace/George-private/testbed;
  sh shell/cpo_27_more.sh 25
"
#screen -r MySessionName0

