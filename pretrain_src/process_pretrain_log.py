import re
import os
import matplotlib.pyplot as plt
import pandas as pd

def parse_log_file(log_file_path):
    """
    解析日志文件，提取验证结果。
    这个修正版本能够正确处理信息分布在不同行的情况。
    """
    validation_results = {}
    current_step = None
    context = None  # 'r2r' or 'rxr'
    task = None     # 'mlm' or 'sap'

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配Step行
            step_match = re.search(r'==============Step (\d+)===============', line)
            if step_match:
                current_step = int(step_match.group(1))
                if current_step not in validation_results:
                    validation_results[current_step] = {}
                context = None
                task = None
                continue

            if current_step is None:
                continue

            # 更新验证类型 (R2R/RXR)
            if 'start validation R2R unseen' in line:
                context = 'r2r'
            elif 'start validation RxR unseen' in line:
                context = 'rxr'

            # 更新验证任务 (MLM/SAP)
            if 'validate val_unseen on mlm task' in line:
                task = 'mlm'
            elif 'validate val_unseen on sap task' in line:
                task = 'sap'

            # 如果找到了 'validation finished' 并且上下文和任务都已确定，则提取结果
            if 'validation finished' in line and context and task:
                if task == 'mlm':
                    acc_match = re.search(r'acc: (\d+\.\d+)', line)
                    if acc_match:
                        key = f'{context}_mlm_acc'
                        validation_results[current_step][key] = float(acc_match.group(1))
                        task = None  # 重置任务状态，避免重复赋值
                elif task == 'sap':
                    gacc_match = re.search(r'gacc: (\d+\.\d+)', line)
                    if gacc_match:
                        key = f'{context}_sap_gacc'
                        validation_results[current_step][key] = float(gacc_match.group(1))
                        task = None  # 重置任务状态

    # 筛选出包含了全部四个指标的完整条目
    complete_results = {
        step: data for step, data in validation_results.items()
        if 'r2r_mlm_acc' in data and 'r2r_sap_gacc' in data and
           'rxr_mlm_acc' in data and 'rxr_sap_gacc' in data
    }
    return complete_results


def calculate_performance(validation_results, mode='total'):
    """
    计算每个checkpoint的性能指标。
    """
    performance_data = []
    for step, data in validation_results.items():
        score = 0
        if mode == 'r2r':
            score = data.get('r2r_mlm_acc', 0) + data.get('r2r_sap_gacc', 0)
        elif mode == 'rxr':
            score = data.get('rxr_mlm_acc', 0) + data.get('rxr_sap_gacc', 0)
        elif mode == 'total':
            score = (data.get('r2r_mlm_acc', 0) + data.get('r2r_sap_gacc', 0) +
                     data.get('rxr_mlm_acc', 0) + data.get('rxr_sap_gacc', 0))
        
        performance_data.append({
            'step': step,
            'r2r_mlm_acc': data.get('r2r_mlm_acc'),
            'r2r_sap_gacc': data.get('r2r_sap_gacc'),
            'rxr_mlm_acc': data.get('rxr_mlm_acc'),
            'rxr_sap_gacc': data.get('rxr_sap_gacc'),
            'score': score
        })
    return sorted(performance_data, key=lambda x: x['score'], reverse=True)


def plot_performance(performance_data, mode, save_path):
    """
    绘制性能变化图。
    """
    df = pd.DataFrame(performance_data).sort_values(by='step')
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df['score'], marker='o', linestyle='-')
    plt.title(f'Performance Trend ({mode.upper()})')
    plt.xlabel('Training Step')
    plt.ylabel('Performance Score')
    plt.grid(True)
    plt.savefig(save_path + 'performance_trend.png')
    plt.close()
    print("性能趋势图已保存为 'performance_trend.png'")


def cleanup_checkpoints(top_checkpoints, ckpts_dir='ckpts'):
    """
    清理除了top 5之外的权重文件。
    """
    if not os.path.isdir(ckpts_dir):
        print(f"找不到 '{ckpts_dir}' 文件夹，跳过清理步骤。")
        return

    top_steps = {ckpt['step'] for ckpt in top_checkpoints}
    all_files = os.listdir(ckpts_dir)
    to_delete = []

    for filename in all_files:
        match = re.match(r'model_step_(\d+)\.pt', filename)
        if match:
            step = int(match.group(1))
            if step not in top_steps:
                to_delete.append(os.path.join(ckpts_dir, filename))

    if not to_delete:
        print("没有需要清理的权重文件。")
        return

    # print("\n以下是将被删除的权重文件：")
    # for f in to_delete:
    #     print(f)
        
    confirm1 = input("你确定要删除其他文件吗？ (yes/no): ").lower()
    if confirm1 == 'yes':
        confirm2 = input("请再次确认，这是一个不可逆操作！ (yes/no): ").lower()
        if confirm2 == 'yes':
            for f in to_delete:
                os.remove(f)
            print("清理完成。")
        else:
            print("操作已取消。")
    else:
        print("操作已取消。")


def main():
    base_path = 'pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/'
    log_path = 'store2/'
    log_file = 'log.txt'
    ckpt_path = 'ckpts'

    # 选择统计模式
    mode = input("请选择统计模式 (r2r, rxr, total): ").lower()
    while mode not in ['r2r', 'rxr', 'total']:
        mode = input("无效的输入，请重新选择 (r2r, rxr, total): ").lower()

    # 核心流程
    validation_results = parse_log_file(base_path + log_path + log_file)
    if not validation_results:
        print("无法从日志文件中解析出任何有效的验证结果。")
        return

    sorted_performance = calculate_performance(validation_results, mode)
    
    top_5_checkpoints = sorted_performance[:5]
    print("\n性能最高的5个Checkpoints:")
    for ckpt in top_5_checkpoints:
        if mode == 'r2r':
            print(f"Step: {ckpt['step']}, r2r_mlm_acc: {ckpt['r2r_mlm_acc']:.2f}, r2r_sap_gacc: {ckpt['r2r_sap_gacc']:.2f}, Score: {ckpt['score']:.2f}")
        elif mode == 'rxr':
            print(f"Step: {ckpt['step']}, rxr_mlm_acc: {ckpt['rxr_mlm_acc']:.2f}, rxr_sap_gacc: {ckpt['rxr_sap_gacc']:.2f}, Score: {ckpt['score']:.2f}")
        else:
            print(f"Step: {ckpt['step']}, r2r_mlm_acc: {ckpt['r2r_mlm_acc']:.2f}, r2r_sap_gacc: {ckpt['r2r_sap_gacc']:.2f}, rxr_mlm_acc: {ckpt['rxr_mlm_acc']:.2f}, rxr_sap_gacc: {ckpt['rxr_sap_gacc']:.2f}, Score: {ckpt['score']:.2f}")

    plot_performance(sorted_performance, mode, base_path + log_path)
    
    cleanup_checkpoints(top_5_checkpoints, base_path + ckpt_path)

if __name__ == '__main__':
    main()