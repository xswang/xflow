#!/bin/bash
rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/pred.txt
rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/pred_0.txt
rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/pred_1.txt
rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/pred_2.txt
Ip=("10.120.15.4" "10.120.15.5")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps
    ssh worker@$ip rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/pred_*.txt
    scp /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps worker@$ip:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/.
done
#/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/dmlc_mpi.py -n 3 -s 1 -H /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/hosts --log-file /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/log.txt /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/data/n2n_train /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/data/n2n_test 
#/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/dmlc_mpi.py -n 3 -s 1 -H /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/hosts --log-file /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/log.txt /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps
/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/dmlc_mpi.py -n 2 -s 3 -H /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/hosts /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps /data12/app_nu_train /data12/app_nu_test
#/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/dmlc_mpi.py -n 1 -s 1 -H /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/hosts --log-file /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/log.txt /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps /data12/app_nu_train_test /data12/app_nu_test
