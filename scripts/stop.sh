ps -ef | grep xflow_lr | awk '{ print $2 }' | sudo xargs kill -9
#ps -ef | grep dump.sh | awk '{ print $2 }' | sudo xargs kill -9
