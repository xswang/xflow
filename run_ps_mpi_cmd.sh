#!/bin/bash
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps
    scp ffm_ps worker@$ip:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/.
done
./dmlc_mpi.py -n 3 -s 3 -H hosts /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/ffm_ps /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/data/u2n_train /home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/data/u2n_test 
