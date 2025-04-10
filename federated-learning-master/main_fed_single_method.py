#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine
import torch
import logging
import os
from datetime import datetime
from collections import Counter

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from models.score_adapted import evaluate_both


def save_metrics_to_excel(metrics, output_dir="./results"):
    """
    将metrics字典保存到Excel文件
    参数:
        metrics: 包含评估指标的嵌套字典
        output_dir: 输出目录路径
    返回:
        excel文件的保存路径
    """
    # 1. 准备数据框
    data = []
    for method, values in metrics.items():
        row = {'Method': method}
        row.update(values)
        data.append(row)

    df = pd.DataFrame(data)

    # 2. 设置列顺序（方法名在第一列）
    columns = ['Method'] + [col for col in df.columns if col != 'Method']
    df = df[columns]

    # 3. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 4. 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_dir, f"metrics_comparison_{timestamp}.xlsx")

    # 5. 保存到Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Metrics')

        # 可选：添加格式美化
        workbook = writer.book
        worksheet = writer.sheets['Metrics']

        # 设置列宽自适应
        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column)
            worksheet.column_dimensions[column[0].column_letter].width = max_length + 2

        # 添加标题样式
        header_row = worksheet[1]
        for cell in header_row:
            cell.font = cell.font.copy(bold=True)

    print(f"Metrics saved to Excel: {excel_path}")
    return excel_path


def setup_logging(dataset, iid, model, score_method):
    # 映射字典
    FV_method_mapping  = {
        1: "True_Shapley",
        2: "MC_Shapley",
        3: "TMC_Shapley",
        4: "GMC_Shapley",
        5: "GTMC_Shapley",
        6: "Random_permuation",
    }
    score_method = FV_method_mapping[score_method]

    # 创建日志目录
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志文件名
    log_file = os.path.join(log_dir,
                            f'log_comparison_between_five_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )


def main_train(args, net_glob, dataset_train, dataset_test, dict_users, method):
    client_choiced = []
    current_num_users = args.num_users
    w_glob = net_glob.state_dict()
    grad_glob = []

    # 初始化评估值数组
    evaluation_values_baseline = np.zeros(current_num_users)
    evaluation_values_MC = np.zeros(current_num_users)
    evaluation_values_TMC = np.zeros(current_num_users)
    evaluation_values_GMC = np.zeros(current_num_users)
    evaluation_values_GTMC = np.zeros(current_num_users)

    # 初始化时间统计变量
    total_train_time = 0.0
    total_eval_time_baseline = 0.0
    total_eval_time_MC = 0.0
    total_eval_time_TMC = 0.0
    total_eval_time_GMC = 0.0
    total_eval_time_GTMC = 0.0  # 修正变量名（原TGMC改为GTMC）
    epoch_times = []

    if args.all_clients:
        w_locals = [copy.deepcopy(w_glob) for _ in range(current_num_users)]
        grads_locals = [copy.deepcopy(w_glob) for _ in range(current_num_users)]

    loss_train = []
    for iter in range(args.epochs):
        epoch_start_time = time.time()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            grads_locals = []

        m = max(int(args.frac * current_num_users), 1)
        idxs_users = np.random.choice(range(current_num_users), m, replace=False)
        client_choiced.append(idxs_users)

        # ===== 训练阶段 =====
        train_start = time.time()
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                grads_locals[idx] = copy.deepcopy(grad)
            else:
                w_locals.append(copy.deepcopy(w))
                grads_locals.append(copy.deepcopy(grad))
            loss_locals.append(copy.deepcopy(loss))
        train_duration = time.time() - train_start
        total_train_time += train_duration

        # ===== 评估阶段 =====
        # Baseline评估
        eval_start = time.time()
        score_round = evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals,
                                    FedAvg(grads_locals), grad_glob, 1)
        eval_duration = time.time() - eval_start
        total_eval_time_baseline += eval_duration
        for user_id in client_choiced[iter]:
            evaluation_values_baseline[user_id] += score_round[user_id]

        # MC评估
        eval_start = time.time()
        score_round = evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals,
                                    FedAvg(grads_locals), grad_glob, 2)
        eval_duration = time.time() - eval_start
        total_eval_time_MC += eval_duration  # 修正：使用MC专用计时变量
        for user_id in client_choiced[iter]:
            evaluation_values_MC[user_id] += score_round[user_id]

        # TMC评估
        eval_start = time.time()
        score_round = evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals,
                                    FedAvg(grads_locals), grad_glob, 3)
        eval_duration = time.time() - eval_start
        total_eval_time_TMC += eval_duration
        for user_id in client_choiced[iter]:
            evaluation_values_TMC[user_id] += score_round[user_id]

        # GMC评估
        eval_start = time.time()
        score_round = evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals,
                                    FedAvg(grads_locals), grad_glob, 4)
        eval_duration = time.time() - eval_start
        total_eval_time_GMC += eval_duration
        for user_id in client_choiced[iter]:
            evaluation_values_GMC[user_id] += score_round[user_id]

        # GTMC评估
        eval_start = time.time()
        score_round = evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals,
                                    FedAvg(grads_locals), grad_glob, 5)
        eval_duration = time.time() - eval_start
        total_eval_time_GTMC += eval_duration  # 修正：使用GTMC专用计时变量
        for user_id in client_choiced[iter]:
            evaluation_values_GTMC[user_id] += score_round[user_id]

        # 记录epoch时间
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        logging.info(
            "epoch: %d | Clients: %s | Train: %.2fs | Eval: %.2fs (B:%.2f/M:%.2f/T:%.2f/G:%.2f/GT:%.2f)",
            iter, list(client_choiced[iter]), train_duration, eval_duration,
            total_eval_time_baseline, total_eval_time_MC, total_eval_time_TMC,
            total_eval_time_GMC, total_eval_time_GTMC
        )

        # 更新全局模型
        w_glob = FedAvg(w_locals)
        grad_glob = FedAvg(grads_locals)
        net_glob.load_state_dict(w_glob)

    # 最终统计输出
    all_clients = np.concatenate(client_choiced)
    count = Counter(all_clients)
    total_eval_time = sum([
        total_eval_time_baseline, total_eval_time_MC,
        total_eval_time_TMC, total_eval_time_GMC,
        total_eval_time_GTMC
    ])

    logging.info("\n===== Final Summary =====")
    logging.info("Total train time: %.2fs", total_train_time)
    logging.info("Total eval time - Baseline: %.2fs", total_eval_time_baseline)
    logging.info("Total eval time - MC: %.2fs", total_eval_time_MC)
    logging.info("Total eval time - TMC: %.2fs", total_eval_time_TMC)
    logging.info("Total eval time - GMC: %.2fs", total_eval_time_GMC)
    logging.info("Total eval time - GTMC: %.2fs", total_eval_time_GTMC)
    logging.info("Avg epoch time: %.2fs", np.mean(epoch_times))
    logging.info("Client participation counts: %s", dict(count.items()))

    # 修正返回值（与调用处的变量名匹配）
    return (
        evaluation_values_baseline, evaluation_values_MC,
        evaluation_values_TMC, evaluation_values_GMC,
        evaluation_values_GTMC,
        total_eval_time_baseline, total_eval_time_MC,
        total_eval_time_TMC, total_eval_time_GMC,
        total_eval_time_GTMC,
        total_train_time
    )


if __name__ == '__main__':
    args = args_parser()
    setup_logging(args.dataset, args.iid, args.model, args.FV_method)
    args.device = torch.device('cuda:{}' if torch.cuda.is_available() else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=False, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=False, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    logging.info(net_glob)
    net_glob.train()


    # baseline_score, teb, ttb = main_train(args, net_glob, dataset_train, dataset_test, dict_users, 1)
    # MC_score, tem, ttm = main_train(args, net_glob, dataset_train, dataset_test, dict_users, 2)
    # TMC_score, tet, ttt = main_train(args, net_glob, dataset_train, dataset_test, dict_users, 3)
    # GMC_score, teg, ttg = main_train(args, net_glob, dataset_train, dataset_test, dict_users, 4)
    # GTMC_score, tetg, tttg = main_train(args, net_glob, dataset_train, dataset_test, dict_users, 5)

    baseline_score, MC_score, TMC_score, GMC_score, GTMC_score, teb, tem, tet, teg, tetg, tt= main_train(args, net_glob, dataset_train, dataset_test, dict_users, 1)
    ttm = ttb = ttt = ttg = tttg = tt

    # 计算MSE, MAE, MRE和余弦相似度

    # 计算评估指标
    metrics = {
        'MC': {
            'time': tem,
            'mse': mean_squared_error(baseline_score, MC_score),
            'mae': mean_absolute_error(baseline_score, MC_score),
            'mre': np.mean(np.abs((MC_score - baseline_score) / (baseline_score + 1e-8))),  # 避免除零
            'cos_sim': 1 - cosine(baseline_score, MC_score)
        },
        'TMC': {
            'time': tet,
            'mse': mean_squared_error(baseline_score, TMC_score),
            'mae': mean_absolute_error(baseline_score, TMC_score),
            'mre': np.mean(np.abs((TMC_score - baseline_score) / (baseline_score + 1e-8))),
            'cos_sim': 1 - cosine(baseline_score, TMC_score)
        },
        'GMC': {
            'time': teg,
            'mse': mean_squared_error(baseline_score, GMC_score),
            'mae': mean_absolute_error(baseline_score, GMC_score),
            'mre': np.mean(np.abs((GMC_score - baseline_score) / (baseline_score + 1e-8))),
            'cos_sim': 1 - cosine(baseline_score, GMC_score)
        },
        'GTMC': {
            'time': tetg,
            'mse': mean_squared_error(baseline_score, GTMC_score),
            'mae': mean_absolute_error(baseline_score, GTMC_score),
            'mre': np.mean(np.abs((GTMC_score - baseline_score) / (baseline_score + 1e-8))),
            'cos_sim': 1 - cosine(baseline_score, GTMC_score)
        }
    }

    save_metrics_to_excel(metrics)

    # 记录到logging
    logging.info("\n===== Evaluation Metrics =====")
    for method, values in metrics.items():
        logging.info(
            f"{method}: Time={values['time']:.2f}s | "
            f"MSE={values['mse']:.4f} | MAE={values['mae']:.4f} | "
            f"MRE={values['mre']:.4f} | CosSim={values['cos_sim']:.4f}"
        )

    # 可视化设置
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    markers = ['o', 's', 'D', '^']  # 不同方法的标记形状
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 对比色

    # 绘制四个子图
    for idx, (metric, title) in enumerate(zip(
            ['mse', 'mae', 'mre', 'cos_sim'],
            ['MSE vs Time', 'MAE vs Time', 'MRE vs Time', 'Cosine Similarity vs Time']
    )):
        ax = axs[idx // 2, idx % 2]
        for i, (method, values) in enumerate(metrics.items()):
            ax.scatter(
                values['time'], values[metric],
                s=100, marker=markers[i], color=colors[i],
                label=f"{method} ({values[metric]:.3f})"  # 在图例中显示数值
            )

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel(metric.upper() if metric != 'cos_sim' else 'Cosine Similarity', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Methods", title_fontsize=10, fontsize=9)

        # # 添加数值标签
        # for method, values in metrics.items():
        #     ax.annotate(
        #         f"{values[metric]:.3f}",
        #         (values['time'], values[metric]),
        #         textcoords="offset points",
        #         xytext=(0, 10), ha='center', fontsize=8
        #     )

    plt.tight_layout(pad=3.0)

    # 保存图像
    output_dir = "./save"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f'metrics_comparison_{timestamp}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Metrics visualization saved to: {file_path}")
