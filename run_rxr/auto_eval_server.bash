export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag_template=" --exp_name release_rxr_grpo
      --run-type eval
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 11
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_rxr_grpo/ckpt.iter{ckpt_num}.pth
      IL.back_algo control
      IL.RECOLLECT_TRAINER.gt_file data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      "

# 初始 ckpt 迭代数字，最终停止的数字
start_ckpt=1500
end_ckpt=1200
step=-20

# 日志文件
log_file="data/logs/checkpoints/release_rxr_grpo/eval_${start_ckpt}-${end_ckpt}.txt"

# 开始循环
for ((ckpt_num=$start_ckpt; ckpt_num>=$end_ckpt; ckpt_num+=$step)); do
    echo "###### eval mode with ckpt.iter${ckpt_num}.pth ######" | tee -a $log_file
    # 将 ckpt_num 替换到 flag 中
    flag=$(echo "$flag_template" | sed "s/{ckpt_num}/$ckpt_num/")
    # 运行 eval 模式
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 2333 run.py $flag 2>&1 | stdbuf -oL grep -v 'it/s' | tee -a $log_file

done

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/auto_eval_server.bash