export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_rxr_dagger
      --run-type dagger
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 6
      ONLY_LAST_SAVEALL True
      IL.iters 30000
      IL.lr 1.5e-5
      IL.min_lr_ratio 0.6
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 5000
      IL.warmup_iters 1000
      IL.load_from_ckpt False
      IL.is_requeue False
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.DATASET.SUFFIX _90
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      IL.expert_policy ndtw
      "

flag2="--exp_name release_rxr_grpo
      --run-type grpo
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 6
      ONLY_LAST_SAVEALL True
      TRAINER_NAME GRPO-R1
      GRPO.iters 1500
      GRPO.lr 2e-5
      GRPO.warmup_iters 0
      GRPO.min_lr_ratio 0.25
      GRPO.log_every 10
      GRPO.load_from_ckpt True
      GRPO.ckpt_to_load data/logs/checkpoints/release_rxr_dagger/store/ckpt.iter20600.pth
      GRPO.is_requeue False
      GRPO.waypoint_aug  True
      GRPO.sample_num 8
      GRPO.update_epochs 1
      GRPO.grpo_beta 0.04
      GRPO.grpo_epsilon 0.2
      GRPO.enable_amp False
      GRPO.enable_all_dropouts True
      GRPO.dropout_in_sampling True
      GRPO.dropout_rate 0.10
      GRPO.max_grad_norm 2.0
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.DATASET.SUFFIX _10
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      "

flag3=" --exp_name release_rxr_grpo
      --run-type eval
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 11
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth
      IL.back_algo control
      IL.RECOLLECT_TRAINER.gt_file data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      "

flag4="--exp_name release_rxr_grpo
      --run-type inference
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth
      INFERENCE.PREDICTIONS_FILE preds.jsonl
      IL.back_algo control
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      "

mode=$1
case $mode in 
      dagger)
      echo "###### dagger train mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag1
      ;;
      grpo)
      echo "###### grpo train mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag2
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag3
      ;;
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag4
      ;;
esac

# 命令行运行：
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main_server.bash dagger 2333
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main_server.bash grpo 2333
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main_server.bash eval 2333
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main_server.bash infer 2333