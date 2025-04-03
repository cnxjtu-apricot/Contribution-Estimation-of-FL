import numpy as np
import itertools
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score
from torchvision import datasets, transforms

from models.Fed import FedAvg
from models.test import test_img
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar

from utils.options import args_parser
from itertools import permutations
import random

import copy
import math


def generate_permutations(num_active_users, args):
    """动态生成排列的核心函数"""
    # 参数解析
    r = 1  # 贡献值最大波动范围
    sigma = 0.01  # 置信度参数
    epsilon = args.Tolerance  # 误差容忍阈值
    N = num_active_users

    # 计算霍夫丁界限
    max_iter_r = (2 * (r ** 2) * N) / (epsilon ** 2) * np.log(2 * N / sigma)
    max_iter_r = int(np.ceil(max_iter_r))

    # 全排列数量计算
    total_perms = math.factorial(num_active_users)

    # 决策逻辑
    if total_perms <= max_iter_r:
        # 全排列模式
        return list(itertools.permutations(range(num_active_users)))
    else:
        # 寻找满足k! >= max_iter_r的最小k
        k = 1
        while k <= num_active_users and math.factorial(k) < max_iter_r:
            k += 1
        k = min(k, num_active_users)  # 确保不超限

        # 生成前k位全排列
        sampled_perms = []
        front_perms = itertools.permutations(range(num_active_users), k)

        # 构建完整排列
        for front in front_perms:
            remaining = list(set(range(num_active_users)) - set(front))
            np.random.shuffle(remaining)
            full_perm = front + tuple(remaining)
            sampled_perms.append(full_perm)

            # 数量控制
            if len(sampled_perms) >= max_iter_r:
                break

        return sampled_perms[:max_iter_r]  # 严格数量控制


# 计算模型F1分数
def calculate_F1(model):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    img_size = dataset_test[0][0].shape
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_test = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_test = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_test = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    all_predictions = []  # 用于保存所有用户的预测
    all_labels = []  # 用于保存所有用户的真实标签

    net_test.load_state_dict(model)

    # 获取本地的预测和真实标签
    # 假设 LocalUpdate 类有这个方法，假设它计算并返回标签
    local = LocalUpdate(args=args, dataset=dataset_test, train=False)
    local_predictions, local_labels = local.test(net=net_test)

    # 将本地的预测和标签加入到列表中
    all_predictions.extend(local_predictions)
    all_labels.extend(local_labels)

    # 计算F1分数
    return f1_score(all_labels, all_predictions, average='weighted')  # 计算加权 F1 分数


# 计算余弦相似度函数
def calculate_cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.view(1, -1), tensor2.view(1, -1)).item()


def calculate_list_cosine_similarity(list1, list2):
    # 遍历字典并计算相似度
    similarities = []
    for i in range(len(list1)):
        similarity = calculate_cosine_similarity(list1[i], list2[i])
        similarities.append(similarity)

    # 计算相似度的均值
    if similarities:
        return sum(similarities) / len(similarities)


def calculate_dict_cosine_similarity(grad_dict1, grad_dict2):
    # 遍历字典并计算相似度
    similarities = []
    for key in grad_dict1.keys():
        if key in grad_dict2:
            similarity = calculate_cosine_similarity(grad_dict1[key], grad_dict2[key])
            similarities.append(similarity)
        else:
            print(f"{key} does not have a corresponding key in yy_grad.")

    # 计算相似度的均值
    if similarities:
        return sum(similarities) / len(similarities)


def evaluate(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    # 使用 get() 来安全地获取方法，如果方法不存在，使用一个默认方法
    method = method_mapping.get(args.FV_method)

    # 调用对应的方法
    result = method(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob)
    return result



def evaluate_both(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob, method):
    # 使用 get() 来安全地获取方法，如果方法不存在，使用一个默认方法
    method = method_mapping.get(method)

    # 调用对应的方法
    result = method(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob)
    return result


def True_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    """修正后的True Shapley实现"""
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)

    # 边界情况处理
    if num_active_users == 0 or len(w_locals) != num_active_users:
        return shapley_values

    try:
        # 生成基于当前活跃用户数量的排列
        permutations = list(itertools.permutations(range(num_active_users)))
    except MemoryError:
        print(f"Too many permutations ({num_active_users} users)")
        return shapley_values

    # 预计算原始用户ID映射
    original_ids = idxs_users  # 直接使用传入的原始ID

    for perm in permutations:
        current_w = copy.deepcopy(w_glob)
        f1_previous = calculate_F1(current_w)

        for i in range(num_active_users):
            # 通过排列索引获取原始用户ID
            user_idx = original_ids[perm[i]]  # 关键修正点

            # 安全边界检查
            if user_idx >= num_total_users:
                continue

            # 合并模型计算
            subset = perm[:i + 1]
            combined_w = [copy.deepcopy(w_locals[j]) for j in subset]
            w_subset = FedAvg(combined_w)

            # 计算贡献
            f1_current = calculate_F1(w_subset)
            contribution = f1_current - f1_previous
            shapley_values[user_idx] += contribution  # 使用原始ID索引
            f1_previous = f1_current

    # 标准化处理
    if len(permutations) > 0:
        shapley_values /= len(permutations)
    return shapley_values


def TMC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)
    tolerance = args.Tolerance

    # 边界情况处理
    if num_active_users == 0 or len(w_locals) != num_active_users:
        return shapley_values

    try:
        # 生成基于当前活跃用户数量的排列
        permutations = list(itertools.permutations(range(num_active_users)))
    except MemoryError:
        print(f"Too many permutations ({num_active_users} users)")
        return shapley_values

    # 预计算原始用户ID映射
    original_ids = idxs_users  # 直接使用传入的原始ID
    current_w = copy.deepcopy(w_glob)
    t = 0  # 排列时间步

    f1_final = calculate_F1(FedAvg(w_locals))

    for perm in permutations:
        t += 1
        f1_previous = calculate_F1(current_w)
        for i in range(num_active_users):
            # 通过排列索引获取原始用户ID
            user_idx = original_ids[perm[i]]  # 关键修正点

            # 安全边界检查
            if user_idx >= num_total_users:
                continue

            if abs(f1_previous - f1_final) <= tolerance:
                shapley_values[user_idx] += 0  # 使用原始ID索引

            else:
                # 合并模型计算
                subset = perm[:i + 1]
                combined_w = [copy.deepcopy(w_locals[j]) for j in subset]
                w_subset = FedAvg(combined_w)

                # 计算贡献
                f1_current = calculate_F1(w_subset)
                contribution = f1_current - f1_previous
                # 动态加权更新Shapley值
                shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution

                # shapley_values[user_idx] += contribution  # 使用原始ID索引，执行静态累加
                f1_previous = f1_current

    # 标准化处理
    shapley_values /= np.sum(shapley_values)
    return shapley_values


def GTG_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    """改进后的GTG Shapley实现"""
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)

    # 有效性校验
    if num_active_users == 0 or len(grads_locals) != num_active_users:
        return shapley_values

    original_ids = idxs_users  # 原始用户ID列表

    try:
        # 方法1：全排列
        # permutations = list(itertools.permutations(range(num_active_users)))

        # 方法2：m-全排列
        permutations = generate_permutations(num_active_users, args)
    except MemoryError:
        print("Permutations memory error!")
        return shapley_values

    t = 0 # 初始化时间步
    for perm in permutations:
        t += 1

        current_g = copy.deepcopy(grad_glob)
        g_score_previous = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0

        for i in range(num_active_users):
            user_idx = original_ids[perm[i]]  # 通过映射获取原始ID

            if user_idx >= num_total_users:
                continue

            # 梯度聚合
            subset = perm[:i + 1]
            combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
            current_g = FedAvg(combined_g)

            # 计算贡献
            g_current = calculate_dict_cosine_similarity(current_g, grad_glob_new)
            contribution = g_current - g_score_previous
            shapley_values[user_idx] += contribution  # 使用原始ID
            shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution

            g_score_previous = g_current

    shapley_values /= np.sum(shapley_values)
    return shapley_values


def Leave_One_Out(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    """改进后的Leave-One-Out实现"""
    num_total_users = args.num_users
    C_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)

    # 空值保护
    if num_active_users < 1:
        return C_values

    original_ids = idxs_users  # 原始用户ID列表

    try:
        # 全集合模型
        all_set_model = FedAvg([copy.deepcopy(w) for w in w_locals])
        f1_best = calculate_F1(all_set_model)
    except Exception as e:
        print(f"LOO failed: {str(e)}")
        return C_values

    for i in range(num_active_users):
        user_id = original_ids[i]  # 当前用户的原始ID

        if user_id >= num_total_users:
            continue

        # 创建排除集（使用本地索引）
        subset = [w for j, w in enumerate(w_locals) if j != i]
        if not subset:
            continue

        try:
            w_subset = FedAvg(subset)
            f1_subset = calculate_F1(w_subset)
            C_values[user_id] = f1_best - f1_subset  # 写入原始ID位置
        except:
            C_values[user_id] = 0

    return C_values


def Random_permuation(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    """随机给出贡献"""
    num_total_users = args.num_users
    C_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)

    # 空值保护
    if num_active_users < 1:
        return C_values

    max_rate = 1
    original_ids = idxs_users  # 原始用户ID列表
    for i in range(num_active_users - 1):
        user_id = original_ids[i]  # 当前用户的原始ID
        C_values[user_id] = random.uniform(0, max_rate)
        max_rate -= C_values[user_id]

    user_id = original_ids[num_active_users - 1]  # 当前用户的原始ID
    C_values[user_id] = max_rate
    return C_values


# 定义一部字典，将方法名映射到相应的函数
method_mapping = {
    1: True_Shapley,
    2: TMC_Shapley,
    3: GTG_Shapley,
    4: Leave_One_Out,
    5: Random_permuation,
}
