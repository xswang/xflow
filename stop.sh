ps -ef | grep ffm_ps | awk '{ print $2 }' | sudo xargs kill -9
