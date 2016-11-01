rm v2v_test-0000*
#python split_data.py test_data_old.txt 3 testdataold
python split_data.py v2v_testdata.txt 3 v2v_test
scp v2v_test-0000* slave1:/home/worker/xiaoshu/logistic-regression-ftrl-ps/data
scp v2v_test-0000* slave2:/home/worker/xiaoshu/logistic-regression-ftrl-ps/data

rm v2v_train-0000*
python split_data.py v2v_train.txt 3 v2v_train
#python split_data.py test_data.txt 3 test_new
scp v2v_train-0000* slave1:/home/worker/xiaoshu/logistic-regression-ftrl-ps/data
scp v2v_train-0000* slave2:/home/worker/xiaoshu/logistic-regression-ftrl-ps/data
