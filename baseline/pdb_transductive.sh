model=$1
dataset=$2
device=$3

[ -z "${model}" ] && model="gat"
[ -z "${dataset}" ] && dataset="cora"
[ -z "${device}" ] && device=-1


python -m pdb src/run.py \
	--model $model \
	--gpu_idx $device \
	--dataset $dataset \
	--task_name transductive