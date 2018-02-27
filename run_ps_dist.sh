#! /bin/sh
#
# run_ps_dist.sh
# Copyright (C) 2018 XiaoshuWang <2012wxs@gmail.com>
#
# Distributed under terms of the MIT license.
#

# if you want run 1 server and 3 workers, each machine lanch one kind of role, you should:
# ########################
 1, on machine 1 which you want lanch server, 
 first run command: sh start_scheduler.sh, this command start scheduler thread
 then run command: sh start_server.sh 1, this command start server thread

 2, on the other 3 machines, run command separately as following:
 sh start_worker 1
# ########################
