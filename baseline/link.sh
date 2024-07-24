model=$1
dataset=$2
device=$3

[ -z "${model}" ] && model="pcfi"
[ -z "${dataset}" ] && dataset="cora"
[ -z "${device}" ] && device=-1



python -m pdb src/run_link.py \
	--model $model \
	--gpu_idx $device \
	--dataset $dataset 