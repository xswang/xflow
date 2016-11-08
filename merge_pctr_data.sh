rm pred.txt
rm pred_0.txt
rm pred_1.txt
rm pred_2.txt
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    scp worker@$ip:/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/pred_*.txt .
done
cat pred_0.txt >> pred.txt
cat pred_1.txt >> pred.txt
cat pred_2.txt >> pred.txt
cp pred.txt ../AUC-caculate-mpi/data/
cd ../AUC-caculate-mpi/data/
sh run_split_data.sh pred.txt
cd ../
sh run.sh
