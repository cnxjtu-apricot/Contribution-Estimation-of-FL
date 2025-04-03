#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import logging
import os
from datetime import datetime
from collections import Counter

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from models.score_adapted import evaluate_both

def setup_logging(dataset, iid, model, score_method):
    # 映射字典
    FV_method_mapping = {
        1: "True Shapley",
        2: "TMC-Shapley",
        3: "GTG-Shapley",
        4: "Leave One Out",
        5: "Random"
    }
    score_method = FV_method_mapping[score_method]

    evaluation_method = "low-quality remove"

    # 创建日志目录
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志文件名
    log_file = os.path.join(log_dir,
                            f'log_{score_method}_{evaluation_method}_{dataset}_{iid}_{model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )


def main_train(args, net_glob, dataset_train, dataset_test, dict_users):
    client_choiced = []
    current_num_users = args.num_users
    w_glob = net_glob.state_dict()
    grad_glob = []
    evaluation_values_f1 = np.zeros(current_num_users)  # 使用当前用户数
    evaluation_values_g = np.zeros(current_num_users)  # 使用当前用户数

    if args.all_clients:
        w_locals = [copy.deepcopy(w_glob) for _ in range(current_num_users)]  # 深拷贝
        grads_locals = [copy.deepcopy(w_glob) for _ in range(current_num_users)]

    loss_train = []
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            grads_locals = []

        m = max(int(args.frac * current_num_users), 1)
        idxs_users = np.random.choice(range(current_num_users), m, replace=False)
        client_choiced.append(idxs_users)

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

        # 计算两种贡献度计算方法的得分
        score_round_f1 = evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals,
                               FedAvg(grads_locals), grad_glob, 1)
        score_round_g = evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals,
                               FedAvg(grads_locals), grad_glob, 3)

        # 登记f1与gtg贡献
        for user_id in client_choiced[iter]:
            evaluation_values_f1[user_id] += score_round_f1[user_id]
            evaluation_values_g[user_id] += score_round_g[user_id]

        # 更新全局模型
        w_glob = FedAvg(w_locals)
        grad_glob = FedAvg(grads_locals)
        net_glob.load_state_dict(w_glob)

    # 统计客户端参与次数
    all_clients = np.concatenate(client_choiced)
    count = Counter(all_clients)
    logging.info("Client participation counts: %s", dict(count.items()))

    # 计算余弦相似度
    cosine_similarity = np.dot(evaluation_values_f1, evaluation_values_g) / (
            np.linalg.norm(evaluation_values_f1) * np.linalg.norm(evaluation_values_g)
    )

    # 计算 MAE
    mae = np.mean(np.abs(evaluation_values_f1 - evaluation_values_g))

    # 计算 MSE
    mse = np.mean((evaluation_values_f1 - evaluation_values_g) ** 2)

    # 输出结果
    print("Cosine Similarity:", cosine_similarity)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)


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
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
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

    main_train(args, net_glob, dataset_train, dataset_test, dict_users)
