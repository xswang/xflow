root_path=`pwd`
echo $root_path
model_name=$1
epochs=$2
sh ./scripts/local.sh 1 3 $root_path/build/test/src/xflow_lr $root_path/data/small_train $root_path/data/small_test $model_name $epochs
