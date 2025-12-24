export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# 代码运行架构：
# 节点：指的是一台独立的物理计算机或服务器，内部可能包含多个gpu。若节点为1则是单机多卡训练，若为1以上，则是多机多卡训练
# 进程(即nproc_per_node的值)：一个“进程”是一个正在运行的程序实例。在 PyTorch 分布式训练（尤其是使用 DistributedDataParallel）中，通常每个 GPU 会由一个独立的 Python 进程来控制和运行训练代码。
# local_rank：当前节点（电脑）下的所有进程（gpu）编号
# world_size/rank：整个分布式训练任务中，参与训练的全局总进程（gpu）数量/编号。对于单机多卡而言，rank就等于local_rank
# 全部代码都运行在一个master节点上；在节点中含有多个进程，每个进程使用一个单独的gpu；
# 每个进程中创建了多个habitat的envs，每个habitat的env包含多个scenes，负责运行该gpu需要运行的episodes中对应scenes的子集
# （torch.distributed.launch默认只能处理一个进程一张gpu的问题,因为训练需要保证每个gpu都至少能单独运行全部可训练权重的网络）


# NUM_ENVIRONMENTS：num_envs_per_gpu
# 只有当IL.load_from_ckpt为True时,IL.is_requeue为True才会起作用;而后者的作用是,改写IL.ckpt_to_load,将其设置为权重文件夹下iteration次数最多的那个权重


flag1="--exp_name release_r2r_dagger
      --run-type dagger
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      IL.iters 30000
      IL.lr 1e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 2000
      IL.warmup_iters 500
      IL.min_lr_ratio 1.0
      IL.load_from_ckpt False
      IL.is_requeue False
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.DATASET.SUFFIX _90
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      "

flag2="--exp_name release_r2r_grpo
      --run-type grpo
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      ONLY_LAST_SAVEALL True
      TRAINER_NAME GRPO-R1
      GRPO.iters 500
      GRPO.lr 2e-5
      GRPO.warmup_iters 0
      GRPO.min_lr_ratio 0.25
      GRPO.log_every 10
      GRPO.load_from_ckpt True
      GRPO.ckpt_to_load data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth
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

flag3=" --exp_name release_r2r_grpo
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth
      IL.back_algo control
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      "

flag4="--exp_name release_r2r_grpo
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth
      INFERENCE.PREDICTIONS_FILE preds.json
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
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash dagger 2333
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash grpo 2333
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash eval 2333
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash infer 2333