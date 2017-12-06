ps -ef | grep lr_ps | awk '{ print $2 }' | sudo xargs kill -9
#ps -ef | grep dump.sh | awk '{ print $2 }' | sudo xargs kill -9
