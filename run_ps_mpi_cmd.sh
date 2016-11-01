#!/bin/bash
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps
done
scp ffm_ps worker@10.101.2.89:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/.
scp ffm_ps worker@10.101.2.90:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/.
#./dmlc_mpi.py -n 3 -s 1 -H hosts /home/worker/xiaoshu/logistic-regression-ftrl-ps/lr_ftrl_ps /home/worker/xiaoshu/logistic-regression-ftrl-ps/data/v2v_train /home/worker/xiaoshu/logistic-regression-ftrl-ps/data/v2v_test 
./dmlc_mpi.py -n 3 -s 3 -H hosts /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/data/ffm_train /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/data/ffm_test 
