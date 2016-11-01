#rm ffmdata_test-0000*
#python split_data.py ffmdata_test.txt 3 ffmdata_test
scp ffm_test-0000* slave1:/home/worker/xiaoshu/Field-aware-Factorization-Machine-mpi/data
scp ffm_test-0000* slave2:/home/worker/xiaoshu/Field-aware-Factorization-Machine-mpi/data

#rm ffmdata_train-0000*
#python split_data.py ffmdata_train.txt 3 ffmdata_train
scp ffm_train-0000* slave1:/home/worker/xiaoshu/Field-aware-Factorization-Machine-mpi/data
scp ffm_train-0000* slave2:/home/worker/xiaoshu/Field-aware-Factorization-Machine-mpi/data
