root_path=`pwd`
echo $root_path
model_name=$1
epochs=$2
#sh ./scripts/local.sh 1 3 $root_path/build/test/src/xflow_lr $root_path/data/small_train $root_path/data/small_test $model_name $epochs

# for test
data_path=/Users/xiaoshuwang/documents/data/libffm_toy
sh ./scripts/local.sh 1 1 $root_path/build/test/src/xflow_lr $data_path/criteo.tr.r100.gbdt0.ffm $data_path/criteo.va.r100.gbdt0.ffm $model_name $epochs
#sh ./scripts/local.sh 1 3 $root_path/build/test/src/xflow_lr $data_path/criteo.tr.r100.gbdt0.ffm $data_path/criteo.va.r100.gbdt0.ffm $model_name $epochs
