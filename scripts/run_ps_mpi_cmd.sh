#!/bin/bash
Ip=("10.120.15.4" "10.120.15.5")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/bin/ffm_ps
    scp /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/bin/ffm_ps worker@$ip:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/bin/.
done
/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/scripts/dmlc_mpi.py -n 2 -s 3 -H /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/scripts/hosts /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/bin/ffm_ps /data12/app_nu_train /data12/app_nu_test
