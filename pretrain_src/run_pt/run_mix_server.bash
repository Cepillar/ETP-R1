
NODE_RANK=0
NUM_GPUS=4
outdir=pretrained/r2r_rxr_ce/mlm.sap_habitat_depth

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
    pretrain_src/pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/mix_model_config_dep.json \
    --config pretrain_src/run_pt/mix_pretrain_server.json \
    --output_dir $outdir

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix_server.bash 2333
