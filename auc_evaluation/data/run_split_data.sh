rm input_data-0000*
python split_data.py $1 3 input_data
scp input_data-0000* slave1:/home/worker/xiaoshu/AUC-caculate-mpi/data
scp input_data-0000* slave2:/home/worker/xiaoshu/AUC-caculate-mpi/data

