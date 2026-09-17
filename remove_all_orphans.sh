# Flower processes — apps first, then the daemons
pkill -f flwr-clientapp
pkill -f flwr-serverapp
pkill -f flower-supernode
pkill -f flower-superlink

# Give them ~5 seconds to exit cleanly, then check
sleep 5
ps -u $USER -o pid,etime,cmd | grep -E 'flwr|flower' | grep -v grep

# Anything still standing gets SIGKILL
pkill -9 -f flwr-
pkill -9 -f flower-

pkill -f "SCREEN"
screen -wipe