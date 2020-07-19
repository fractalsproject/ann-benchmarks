#!/bin/bash

SESH=`uuidgen`

echo 'Launching screen session=$SESH'
screen -dmS $SESH -L -Logfile "$SESH.screenlog"
screen -S $SESH -X colon "logfile flush 0^M"

# optionally go to session
echo 'Joining new screen session=$SESH'
sleep 1
screen -r $SESH

# or follow the log ?
# tail -Fn 0 my.log
