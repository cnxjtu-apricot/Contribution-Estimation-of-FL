# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Python version: 3.6
#
# import matplotlib
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import copy
# import numpy as np
# from torchvision import datasets, transforms
# import torch
# import logging
# import os
# from datetime import datetime
# from collections import Counter
#
# from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
# from utils.options import args_parser
# from models.Update import LocalUpdate
# from models.Nets import MLP, CNNMnist, CNNCifar
# from models.Fed import FedAvg
# from models.test import test_img
# from models.score import evaluate
#
#
# def p_num(n):
#     result = 1
#     while n > 1:
#         result *= n
#         n -= 1
#     return result
#
#
# def setup_logging(dataset, iid, model, score_method):
#     # 映射字典
#     FV_method_mapping = {
#         1: "True Shapley",
#         2: "TMC-Shapley",
#         3: "GTG-Shapley",
#         4: "Leave One Out",
#         5: "Random"
#     }
#     score_method = FV_method_mapping[score_method]
#
#     evaluation_method = "low-quality remove"
#
#     # 创建日志目录
#     log_dir = './log'
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
#     # 设置日志文件名
#     log_file = os.path.join(log_dir,
#                             f'log_{score_method}_{evaluation_method}_{dataset}_{iid}_{model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
#
#     # 配置日志记录
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),  # 输出到文件
#             logging.StreamHandler()  # 输出到控制台
#         ]
#     )
#
#
# def main_train(args, net_glob, dataset_train, dataset_test, dict_users, current_num_users):
#     client_choiced = []  # 改为局部变量
#     w_glob = net_glob.state_dict()
#     grad_glob = []
#     evaluation_values = np.zeros(current_num_users)  # 使用当前用户数
#
#     if args.all_clients:
#         w_locals = [copy.deepcopy(w_glob) for _ in range(current_num_users)]  # 深拷贝
#         grads_locals = [copy.deepcopy(w_glob) for _ in range(current_num_users)]
#
#     loss_train = []
#     for iter in range(args.epochs):
#         loss_locals = []
#         if not args.all_clients:
#             w_locals = []
#             grads_locals = []
#
#         m = max(int(args.frac * current_num_users), 1)
#         idxs_users = np.random.choice(range(current_num_users), m, replace=False)
#         client_choiced.append(idxs_users)
#
#         for idx in idxs_users:
#             local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
#             w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))
#
#             if args.all_clients:
#                 w_locals[idx] = copy.deepcopy(w)
#                 grads_locals[idx] = copy.deepcopy(grad)
#             else:
#                 w_locals.append(copy.deepcopy(w))
#                 grads_locals.append(copy.deepcopy(grad))
#             loss_locals.append(copy.deepcopy(loss))
#
#         # 贡献值累加修正
#         score_round = evaluate(args, w_locals, idxs_users, w_glob, grads_locals,
#                                FedAvg(grads_locals), grad_glob)
#         for user_idx, score in zip(idxs_users, score_round):  # 关键修改
#             evaluation_values[user_idx] += score
#
#         # 更新全局模型
#         w_glob = FedAvg(w_locals)
#         grad_glob = FedAvg(grads_locals)
#         net_glob.load_state_dict(w_glob)
#
#     # 统计客户端参与次数
#     all_clients = np.concatenate(client_choiced)
#     count = Counter(all_clients)
#     logging.info("Client participation counts: %s", dict(count.items()))
#
#     return evaluation_values.tolist()
#
#
# def evaluate_contribution_and_remove_users(args, net_glob_origin, dataset_train, dataset_test, dict_users):
#     user_ids = list(range(args.num_users))
#     remaining_users = user_ids.copy()
#     original_num_users = args.num_users
#     flag = 1 # 标记是否未删除
#     acc_history = []
#
#     while len(remaining_users) > 1:
#         # 调整当前用户数
#         current_num_users = len(remaining_users)
#         current_dict = {i: dict_users[uid] for i, uid in enumerate(remaining_users)}
#
#         # 使用深拷贝的初始模型
#         net_glob = copy.deepcopy(net_glob_origin)
#         args.num_users = current_num_users  # 临时修改参数
#
#         # 训练并获取贡献值
#         evaluation_values = main_train(
#             args, net_glob, dataset_train, dataset_test,
#             current_dict, current_num_users
#         )
#
#         # 恢复原始用户数参数
#         args.num_users = original_num_users
#
#         if flag:
#             logging.info(
#                 f"Origin Test Accuracy: {acc_test:.2f}%")
#             flag = 0
#         else:
#             logging.info(f"After removing user {remove_user}, Remaining: {remaining_users}, Test Accuracy: {acc_test:.2f}%")
#
#
#         # 排序并删除最高贡献用户
#         sorted_users = sorted(
#             zip(remaining_users, evaluation_values),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         remove_user = sorted_users[0][0]
#         remaining_users.remove(remove_user)
#
#         # 测试准确率
#         acc_test, _ = test_img(net_glob, dataset_test, args)
#         acc_history.append(acc_test)
#
#     return acc_history
#
#
# if __name__ == '__main__':
#     args = args_parser()
#     setup_logging(args.dataset, args.iid, args.model, args.FV_method)
#     args.device = torch.device('cuda:{}' if torch.cuda.is_available() else 'cpu')
#
#     # load dataset and split users
#     if args.dataset == 'mnist':
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
#         dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
#         # sample users
#         if args.iid:
#             dict_users = mnist_iid(dataset_train, args.num_users)
#         else:
#             dict_users = mnist_noniid(dataset_train, args.num_users)
#     elif args.dataset == 'cifar':
#         trans_cifar = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
#         dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
#         if args.iid:
#             dict_users = cifar_iid(dataset_train, args.num_users)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')
#     else:
#         exit('Error: unrecognized dataset')
#     img_size = dataset_train[0][0].shape
#
#     # build model
#     if args.model == 'cnn' and args.dataset == 'cifar':
#         net_glob = CNNCifar(args=args).to(args.device)
#     elif args.model == 'cnn' and args.dataset == 'mnist':
#         net_glob = CNNMnist(args=args).to(args.device)
#     elif args.model == 'mlp':
#         len_in = 1
#         for x in img_size:
#             len_in *= x
#         net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
#     else:
#         exit('Error: unrecognized model')
#
#     logging.info(net_glob)
#     net_glob.train()
#
#     acc_history = evaluate_contribution_and_remove_users(args, net_glob, dataset_train, dataset_test, dict_users)
#
#     plt.figure()
#     plt.plot(range(len(acc_history)), acc_history)
#
#     FV_method_mapping = {
#         1: "True Shapley",
#         2: "TMC-Shapley",
#         3: "GTG-Shapley",
#         4: "Leave One Out",
#         5: "Random"
#     }
#
#     plt.savefig(f'./save/fed_{args.dataset}_method{FV_method_mapping[args.FV_method]}.png')

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
# from models.score import evaluate
from models.score_adapted import evaluate


def setup_logging(dataset, iid, model, score_method):
    FV_method_mapping = {
        1: "True Shapley",
        2: "TMC-Shapley",
        3: "GTG-Shapley",
        4: "Leave One Out",
        5: "Random"
    }
    score_method = FV_method_mapping[score_method]
    evaluation_method = "low-quality remove"

    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir,
                            f'log_{score_method}_{evaluation_method}_{dataset}_{iid}_{model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def main_train(args, net_glob, dataset_train, dict_users, user_mask):
    """ 总体训练流程，不设掩膜，用户随机生成，获得初始子集 """
    client_choiced = [] # 记录每轮的参与者子集
    w_glob = net_glob.state_dict()
    grad_glob = {key: torch.zeros_like(value) for key, value in w_glob.items()}
    evaluation_values = np.zeros(args.num_users)

    active_users = np.where(user_mask == False)[0]  # 只考虑活跃用户
    current_num_users = len(active_users)

    # # 方案二初始化
    # client_participation = {}  # 记录客户端是否参与上一轮训练
    # client_prob = {}  # 记录客户端的参与概率
    # initial_prob = 1.0 / args.epochs  # 初始参与概率

    # for user in active_users:
    #     client_participation[user] = False  # 初始都未参与
    #     client_prob[user] = initial_prob  # 初始概率

    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        grads_locals = []

        # 方案一：只从活跃用户中随机选择一定比例的用户
        m = max(int(args.frac * current_num_users), 1)
        idxs_users = np.random.choice(active_users, m, replace=False)
        client_choiced.append(idxs_users)

        # # 方案二：动态调整参与者，参与者数量最多不超过frac * current_num_users
        # if iter == 0:
        #     # 第一轮：随机选择初始参与者
        #     m = max(int(args.frac * current_num_users), 1)
        #     idxs_users = np.random.choice(active_users, m, replace=False)
        #     client_choiced.append(idxs_users)
        #
        #     # 更新参与状态
        #     for user in active_users:
        #         client_participation[user] = (user in idxs_users)
        # else:
        #     # 后续轮次：动态调整参与者
        #     new_participants = []
        #     num_new_participation = 0
        #     for user in active_users:
        #         if client_participation[user]:
        #             # 上一轮参与者：以 prob 的概率退出
        #             if np.random.rand() < client_prob[user] | new_participants >= args.frac * current_num_users:
        #                 client_prob[user] = 1.0 / args.epochs  # 退出，不参与本轮, 重置概率
        #             else:
        #                 new_participants.append(user)  # 继续参与
        #                 num_new_participation += 1
        #                 client_prob[user] *= 2  # 未退出，概率翻倍
        #         else:
        #             # 上一轮未参与者：以 prob 的概率加入
        #             if np.random.rand() < client_prob[user] & new_participants < args.frac * current_num_users:
        #                 new_participants.append(user)  # 加入本轮
        #                 num_new_participation += 1
        #                 client_prob[user] = 1.0 / args.epochs  # 加入后退出概率归零
        #             else:
        #                 client_prob[user] *= 2  # 未加入，概率翻倍
        #
        #     # 确保至少有一个参与者
        #     if len(new_participants) == 0:
        #         m = max(int(args.frac * current_num_users), 1)
        #         new_participants = np.random.choice(active_users, m, replace=False).tolist()
        #
        #     idxs_users = np.array(new_participants)
        #     client_choiced.append(idxs_users)
        #
        #     # 更新参与状态
        #     for user in active_users:
        #         client_participation[user] = (user in idxs_users)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            grads_locals.append(copy.deepcopy(grad))
            loss_locals.append(copy.deepcopy(loss))

        # 贡献值累加（直接使用原始ID）
        score_round = evaluate(args, w_locals, idxs_users, w_glob, grads_locals,
                               FedAvg(grads_locals), grad_glob)

        # 更新全局模型
        w_glob = FedAvg(w_locals)
        grad_glob = FedAvg(grads_locals)
        net_glob.load_state_dict(w_glob)

        for user_id in idxs_users:
            evaluation_values[user_id] += score_round[user_id]

        logging.info("epoch: %s, Clients participation: %s", iter, list(client_choiced[iter]))
        logging.info("Clients Score: %s", list(evaluation_values))

    # 统计时直接使用原始ID
    all_clients = np.concatenate(client_choiced)
    count = Counter(all_clients)
    logging.info("Active clients participation: %s", dict(count.items()))

    return evaluation_values, client_choiced

def removed_train(args, net_glob, dataset_train, dict_users, user_mask, client_choiced):
    """ 移除用户训练函数， 每次从已有参与者子集中移除指定的用户 """
    w_glob = net_glob.state_dict()
    grad_glob = {key: torch.zeros_like(value) for key, value in w_glob.items()}
    evaluation_values = np.zeros(args.num_users)

    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        grads_locals = []

        # 参与者为曾经的对应轮次子集
        for idx in client_choiced[iter]:
            # 如果用户被移除，跳过他
            if user_mask[idx]:
                client_choiced[iter] = np.delete(client_choiced[iter], np.where(client_choiced[iter] == idx))
                continue

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            grads_locals.append(copy.deepcopy(grad))
            loss_locals.append(copy.deepcopy(loss))

        # 贡献值累加（直接使用原始ID）
        if len(grads_locals) == 0:
            logging.info("epoch: %s, no participant", iter)
            logging.info("remained no change")
            continue

        else:
            score_round = evaluate(args, w_locals, client_choiced[iter], w_glob, grads_locals,
                                    FedAvg(grads_locals), grad_glob)

        # 更新全局模型
        w_glob = FedAvg(w_locals)
        grad_glob = FedAvg(grads_locals)
        net_glob.load_state_dict(w_glob)

        for user_id in client_choiced[iter]:
            evaluation_values[user_id] += score_round[user_id]

        logging.info("epoch: %s, Clients participation: %s", iter, list(client_choiced[iter]))
        logging.info("Clients Score: %s", list(evaluation_values))

    # 统计时直接使用原始ID
    all_clients = np.concatenate(client_choiced)
    count = Counter(all_clients)
    logging.info("Active clients participation: %s", dict(count.items()))

    return evaluation_values, client_choiced


def evaluate_with_mask(args, net_glob_origin, dataset_train, dataset_test, dict_users):
    user_mask = np.zeros(args.num_users, dtype=bool)  # False=活跃
    acc_record = []

    inner_net_glob = copy.deepcopy(net_glob_origin)

    # 训练并获取贡献值
    evaluation_values, subset_list = main_train(
        args, inner_net_glob, dataset_train,
        dict_users, user_mask
    )

    # 测试准确率
    acc_test, _ = test_img(inner_net_glob, dataset_test, args)
    acc_record.append(acc_test)

    logging.info(f"Deactivated user -1, "
                 f"Remaining active: {np.sum(~user_mask)}, "
                 f"Test Accuracy: {acc_test:.2f}%")

    while np.sum(~user_mask) > 2:  # 当活跃用户>2时继续
        # 找出活跃用户中贡献最高的标记为不活跃
        active_ids = np.where(~user_mask)[0]  # where返回了包含一个满足条件的数组的数组 [[0,1,2...]]
        highest_contrib = active_ids[np.argmax([evaluation_values[i] for i in active_ids])]
        user_mask[highest_contrib] = True  # 标记为不活跃

        inner_net_glob = copy.deepcopy(net_glob_origin)

        # 训练并获取贡献值
        evaluation_values, subset_list = removed_train(
            args, inner_net_glob, dataset_train,
            dict_users, user_mask, subset_list
        )

        # 测试准确率
        acc_test, _ = test_img(inner_net_glob, dataset_test, args)
        acc_record.append(acc_test)

        logging.info(f"Deactivated user {highest_contrib}, "
                     f"Remaining active: {np.sum(~user_mask)}, "
                     f"Test Accuracy: {acc_test:.2f}%")

    return acc_record


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

    acc_history = evaluate_with_mask(args, net_glob, dataset_train, dataset_test, dict_users)

    # 可视化结果保存
    FV_method_mapping = {1: "True Shapley", 2: "TMC-Shapley", 3: "GTG-Shapley", 4: "Leave One Out", 5: "Random"}
    plt.figure(figsize=(10, 6))
    plt.plot(acc_history, marker='o')
    plt.title(f"Accuracy Trend ({FV_method_mapping[args.FV_method]})")
    plt.xlabel("Removal Steps")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.savefig(f'./save/fed_{args.dataset}_method{FV_method_mapping[args.FV_method]}.png', bbox_inches='tight')
    plt.close()
