import numpy as np
import itertools
import torch
import torch.nn.functional as F
import datetime
import sys
from functools import wraps

from sklearn.metrics import accuracy_score
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

# 带正负的归一化
def signed_normalize(shapley_values, idxs_users):
    """
    保留正负信息的归一化方法
    参数：
        shapley_values: 原始Shapley值数组（含正负值）
        idxs_users: 需要归一化的用户索引
    返回：
        归一化后的Shapley值（范围[-1,1]）
    """
    # 分离正负部分
    pos_values = np.where(shapley_values > 0, shapley_values, 0)
    neg_values = np.where(shapley_values < 0, -shapley_values, 0)  # 取绝对值

    # 分别计算正负和
    sum_pos = np.sum(pos_values[idxs_users])
    sum_neg = np.sum(neg_values[idxs_users])

    # 分别归一化
    normalized = np.zeros_like(shapley_values)
    for idx in idxs_users:
        if shapley_values[idx] > 0:
            normalized[idx] = shapley_values[idx] / (sum_pos + 1e-10)  # 防止除零
        else:
            normalized[idx] = shapley_values[idx] / (sum_neg + 1e-10)

    return normalized
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
        if args.FV_method >= 4:
            # 评估方案需要进行引导
            # 寻找满足k! >= max_iter_r的最小k
            k = 1
            while k <= num_active_users and math.factorial(num_active_users)/math.factorial(num_active_users - k) < max_iter_r:
                k += 1
            k = min(k - 1, num_active_users)  # 确保不超限

            # 生成前k位全排列P(n,k)
            sampled_perms = []

            # 构建完整排列
            while len(sampled_perms) < max_iter_r:
                front_perms = itertools.permutations(range(num_active_users), k)
                for front in front_perms:
                    remaining = list(set(range(num_active_users)) - set(front))
                    np.random.shuffle(remaining)
                    full_perm = front + tuple(remaining)
                    sampled_perms.append(full_perm)

                    # 数量控制
                    if len(sampled_perms) >= max_iter_r:
                        break

        else:
            # 不需要引导，随机抽样max_iter_r
            all_perms = list(itertools.permutations(range(num_active_users), num_active_users))
            sampled_perms = random.sample(all_perms, min(max_iter_r, len(all_perms)))  # 确保不超限


        return sampled_perms[:max_iter_r]  # 严格数量控制

# 计算余弦相似度函数
def calculate_cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.view(1, -1), tensor2.view(1, -1)).item()


def calculate_mse(tensor1, tensor2):
    return F.mse_loss(tensor1, tensor2).item()

def calculate_list_cosine_similarity(list1, list2):
    # 遍历字典并计算相似度
    args = args_parser()
    similarities = []
    for i in range(len(list1)):
        if args.v_func == 0:
            similarity = calculate_cosine_similarity(list1[i], list2[i])
        else:
            similarity = calculate_mse(list1[i], list2[i])
        similarities.append(similarity)

    # 计算相似度的均值
    if similarities:
        return sum(similarities) / len(similarities)


def calculate_dict_cosine_similarity(grad_dict1, grad_dict2):
    # 遍历字典并计算相似度'
    args = args_parser()
    similarities = []
    for key in grad_dict1.keys():
        if key in grad_dict2:
            if args.v_func == 0:
                similarity = calculate_cosine_similarity(grad_dict1[key], grad_dict2[key])
            else:
                similarity = calculate_mse(grad_dict1[key], grad_dict2[key])
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



# def True_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
#     """True Shapley实现"""
#     num_total_users = args.num_users
#     shapley_values = np.zeros(num_total_users)
#     num_active_users = len(idxs_users)
#
#     # 有效性校验
#     if num_active_users == 0 or len(grads_locals) != num_active_users:
#         return shapley_values
#
#     original_ids = idxs_users  # 原始用户ID列表
#
#     try:
#         # 全排列计算
#         permutations = list(itertools.permutations(range(num_active_users)))
#     except MemoryError:
#         print("Permutations memory error!")
#         return shapley_values
#
#     t = 0 # 初始化时间步
#
#     for perm in permutations:
#         t += 1
#
#         current_g = copy.deepcopy(grad_glob)
#         g_score_previous = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0
#
#         for i in range(num_active_users):
#             user_idx = original_ids[perm[i]]  # 通过映射获取原始ID
#
#             if user_idx >= num_total_users:
#                 continue
#
#             # 梯度聚合
#             subset = perm[:i + 1]
#             combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
#             current_g = FedAvg(combined_g)
#
#             # 计算贡献
#             g_current = calculate_dict_cosine_similarity(current_g, grad_glob_new)
#             contribution = g_current - g_score_previous
#             shapley_values[user_idx] += contribution  # 使用原始ID
#             shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution
#
#             g_score_previous = g_current
#
#
#         # if t == 1000:
#         #     current_time = datetime.datetime.now()
#         #     print("当前时间:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
#
#     shapley_values /= np.sum(shapley_values)
#     return shapley_values


# def MC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
#     """改进后的GTG Shapley实现"""
#     num_total_users = args.num_users
#     shapley_values = np.zeros(num_total_users)
#     num_active_users = len(idxs_users)
#
#     # 有效性校验
#     if num_active_users == 0 or len(grads_locals) != num_active_users:
#         return shapley_values
#
#     original_ids = idxs_users  # 原始用户ID列表
#
#     try:
#         # 从全排列中随机抽样(MC)
#         permutations = generate_permutations(num_active_users, args)
#     except MemoryError:
#         print("Permutations memory error!")
#         return shapley_values
#
#     t = 0 # 初始化时间步
#     for perm in permutations:
#         t += 1
#
#         current_g = copy.deepcopy(grad_glob)
#         g_score_previous = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0
#
#         for i in range(num_active_users):
#             user_idx = original_ids[perm[i]]  # 通过映射获取原始ID
#
#             if user_idx >= num_total_users:
#                 continue
#
#             # 梯度聚合
#             subset = perm[:i + 1]
#             combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
#             current_g = FedAvg(combined_g)
#
#             # 计算贡献
#             g_current = calculate_dict_cosine_similarity(current_g, grad_glob_new)
#             contribution = g_current - g_score_previous
#             shapley_values[user_idx] += contribution  # 使用原始ID
#             shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution
#
#             g_score_previous = g_current
#
#     shapley_values /= np.sum(shapley_values)
#     return shapley_values
#
# def TMC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
#     """改进后的GTG Shapley实现"""
#     num_total_users = args.num_users
#     shapley_values = np.zeros(num_total_users)
#     num_active_users = len(idxs_users)
#
#     # 有效性校验
#     if num_active_users == 0 or len(grads_locals) != num_active_users:
#         return shapley_values
#
#     original_ids = idxs_users  # 原始用户ID列表
#
#     try:
#         # MC抽样全排列
#         permutations_left = generate_permutations(num_active_users, args)
#     except MemoryError:
#         print("Permutations memory error!")
#         return shapley_values
#
#     t = 0 # 初始化时间步
#     # 初始化收敛判断所需的历史沙普利值记录
#     shapley_history = []  # 用于存储最近10次迭代的沙普利值
#     converged = False
#
#     # 带轮间截断
#     while not converged:
#         # 随机采样一个排列并移除
#         perm = random.choice(permutations_left)
#         permutations_left.remove(perm)
#         t += 1
#
#         # 计算沙普利值
#         current_g = copy.deepcopy(grad_glob)
#         g_score_previous = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0
#
#         for i in range(num_active_users):
#             user_idx = original_ids[perm[i]]
#             if user_idx >= num_total_users:
#                 continue
#
#             # 梯度聚合与贡献计算
#             subset = perm[:i + 1]
#             combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
#             current_g = FedAvg(combined_g)
#
#             g_current = calculate_dict_cosine_similarity(current_g, grad_glob_new)
#             contribution = g_current - g_score_previous
#
#             shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution
#
#             # 轮内截断
#             if math.fabs(contribution) < args.Tolerance:
#                 break
#
#             g_score_previous = g_current
#
#         # 收敛条件判断（每迭代1次执行）
#         shapley_history.append(shapley_values.copy())
#         if len(shapley_history) > 10:
#             shapley_history.pop(0)  # 保持最近10次记录
#
#         if len(shapley_history) == 10:
#             # 计算相对变化率（公式10）
#             delta_sum = 0
#             for i in range(num_total_users):
#                 current_val = shapley_history[-1][i]
#                 if abs(current_val) > 1e-6:  # 避免除零
#                     delta_sum += np.mean([abs(shapley_history[-1][i] - shapley_history[k][i]) / abs(current_val)
#                                           for k in range(9)])
#
#             avg_delta = delta_sum / num_total_users
#             converged = (avg_delta < 0.05)  # 阈值条件
#
#     shapley_values /= np.sum(shapley_values)
#     return shapley_values
#
# def GMC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
#     """改进后的GTG Shapley实现"""
#     num_total_users = args.num_users
#     shapley_values = np.zeros(num_total_users)
#     num_active_users = len(idxs_users)
#
#     # 有效性校验
#     if num_active_users == 0 or len(grads_locals) != num_active_users:
#         return shapley_values
#
#     original_ids = idxs_users  # 原始用户ID列表
#
#     try:
#         # m-全排列
#         permutations = generate_permutations(num_active_users, args)
#     except MemoryError:
#         print("Permutations memory error!")
#         return shapley_values
#
#     t = 0 # 初始化时间步
#     for perm in permutations:
#         t += 1
#
#         current_g = copy.deepcopy(grad_glob)
#         g_score_previous = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0
#
#         for i in range(num_active_users):
#             user_idx = original_ids[perm[i]]  # 通过映射获取原始ID
#
#             if user_idx >= num_total_users:
#                 continue
#
#             # 梯度聚合
#             subset = perm[:i + 1]
#             combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
#             current_g = FedAvg(combined_g)
#
#             # 计算贡献
#             g_current = calculate_dict_cosine_similarity(current_g, grad_glob_new)
#             contribution = g_current - g_score_previous
#             shapley_values[user_idx] += contribution  # 使用原始ID
#             shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution
#
#             g_score_previous = g_current
#
#     shapley_values /= np.sum(shapley_values)
#     return shapley_values
#
# def GTMC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
#     """改进后的GTG Shapley实现"""
#     num_total_users = args.num_users
#     shapley_values = np.zeros(num_total_users)
#     num_active_users = len(idxs_users)
#
#     # 有效性校验
#     if num_active_users == 0 or len(grads_locals) != num_active_users:
#         return shapley_values
#
#     original_ids = idxs_users  # 原始用户ID列表
#
#     try:
#         # m-全排列
#         permutations_left = generate_permutations(num_active_users, args)
#     except MemoryError:
#         print("Permutations memory error!")
#         return shapley_values
#
#     t = 0 # 初始化时间步
#
#     # 初始化收敛判断所需的历史沙普利值记录
#     shapley_history = []  # 用于存储最近10次迭代的沙普利值
#     converged = False
#
#     # 带轮间截断
#     while not converged:
#         # 随机采样一个排列并移除
#         perm = random.choice(permutations_left)
#         permutations_left.remove(perm)
#         t += 1
#
#         # 计算沙普利值
#         current_g = copy.deepcopy(grad_glob)
#         g_score_previous = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0
#
#         for i in range(num_active_users):
#             user_idx = original_ids[perm[i]]
#             if user_idx >= num_total_users:
#                 continue
#
#             # 梯度聚合与贡献计算
#             subset = perm[:i + 1]
#             combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
#             current_g = FedAvg(combined_g)
#
#             g_current = calculate_dict_cosine_similarity(current_g, grad_glob_new)
#             contribution = g_current - g_score_previous
#
#             shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution
#
#             # 轮内截断
#             if math.fabs(contribution) < args.Tolerance:
#                 break
#
#             g_score_previous = g_current
#
#         # 收敛条件判断（每迭代1次执行）
#         shapley_history.append(shapley_values.copy())
#         if len(shapley_history) > 10:
#             shapley_history.pop(0)  # 保持最近10次记录
#
#         if len(shapley_history) == 10:
#             # 计算相对变化率（公式10）
#             delta_sum = 0
#             for i in range(num_total_users):
#                 current_val = shapley_history[-1][i]
#                 if abs(current_val) > 1e-6:  # 避免除零
#                     delta_sum += np.mean([abs(shapley_history[-1][i] - shapley_history[k][i]) / abs(current_val)
#                                           for k in range(9)])
#
#             avg_delta = delta_sum / num_total_users
#             converged = (avg_delta < 0.05)  # 阈值条件
#
#     shapley_values /= np.sum(shapley_values)
#     return shapley_values


''' 加入了dp保存Fedavg结果，防止重复计算 '''
class ShapleyCache:
    """DP缓存工具类"""
    def __init__(self):
        self.grad_cache = {}  # {frozenset(subset): (agg_grad, similarity)}

    def get_or_compute(self, subset, grads_locals, grad_glob_new):
        key = frozenset(subset)
        if key in self.grad_cache:
            return self.grad_cache[key]

        combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
        agg_grad = FedAvg(combined_g)
        similarity = calculate_dict_cosine_similarity(agg_grad, grad_glob_new)

        self.grad_cache[key] = (agg_grad, similarity)
        return agg_grad, similarity


def True_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    """
        优化后的True Shapley实现（带DP缓存）
        注意这里的grad_glob_new仅作为基线，即测试子集与该基线的相似度
    """
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)

    if num_active_users == 0 or len(grads_locals) != num_active_users:
        return shapley_values

    original_ids = idxs_users
    cache = {}  # DP缓存：{subset_tuple: (aggregated_grad, similarity_score)}

    try:
        permutations = list(itertools.permutations(range(num_active_users)))
    except MemoryError:
        print("Permutations memory error!")
        return shapley_values

    t = 0
    current_g = copy.deepcopy(grad_glob)
    g_score_previous_origin = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0

    for perm in permutations:
        t += 1
        g_score_previous = g_score_previous_origin

        for i in range(num_active_users):
            user_idx = original_ids[perm[i]]
            if user_idx >= num_total_users:
                continue

            subset = tuple(sorted(perm[:i + 1]))  # 转换为可哈希的排序元组

            # 检查缓存
            if subset in cache:
                current_g, g_current = cache[subset]
            else:
                # 未命中则计算并缓存
                combined_g = [copy.deepcopy(grads_locals[j]) for j in subset]
                current_g = FedAvg(combined_g)
                g_current = calculate_dict_cosine_similarity(current_g, grad_glob_new)
                cache[subset] = (current_g, g_current)

            # 贡献计算
            contribution = g_current - g_score_previous
            shapley_values[user_idx] = (t - 1) / t * shapley_values[user_idx] + (1 / t) * contribution
            g_score_previous = g_current

        # if t % 10000 == 0:
        #     current_time = datetime.datetime.now()
        #     print("当前时间:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

    # softmax方案
    # sum_shapley = 0
    # for idx in idxs_users:
    #     shapley_values[idx] = np.exp(shapley_values[idx])
    #     sum_shapley += shapley_values[idx]
    # for idx in idxs_users:
    #     shapley_values[idx] /= sum_shapley
    shapley_values = signed_normalize(shapley_values, idxs_users)

    return shapley_values

def MC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)
    original_ids = idxs_users
    cache = ShapleyCache()  # 初始化DP缓存

    if num_active_users == 0 or len(grads_locals) != num_active_users:
        return shapley_values

    try:
        permutations = generate_permutations(num_active_users, args)
    except MemoryError:
        print("Permutations memory error!")
        return shapley_values

    t = 0
    current_g = copy.deepcopy(grad_glob)
    g_score_previous_origin = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0

    for perm in permutations:
        t += 1
        g_score_previous = g_score_previous_origin

        for i in range(num_active_users):
            user_idx = original_ids[perm[i]]
            if user_idx >= num_total_users:
                continue

            subset = perm[:i+1]
            current_g, g_current = cache.get_or_compute(subset, grads_locals, grad_glob_new)  # 使用缓存

            contribution = g_current - g_score_previous
            shapley_values[user_idx] = (t-1)/t * shapley_values[user_idx] + (1/t) * contribution
            g_score_previous = g_current

    # softmax方案
    # sum_shapley = 0
    # for idx in idxs_users:
    #     shapley_values[idx] = np.exp(shapley_values[idx])
    #     sum_shapley += shapley_values[idx]
    # for idx in idxs_users:
    #     shapley_values[idx] /= sum_shapley

    shapley_values = signed_normalize(shapley_values, idxs_users)

    return shapley_values

def TMC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)
    original_ids = idxs_users
    cache = ShapleyCache()  # 初始化DP缓存

    if num_active_users == 0 or len(grads_locals) != num_active_users:
        return shapley_values

    try:
        permutations_left = generate_permutations(num_active_users, args)
    except MemoryError:
        print("Permutations memory error!")
        return shapley_values

    t, converged = 0, False
    shapley_history = []
    current_g = copy.deepcopy(grad_glob)
    g_score_previous_origin = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0

    while not converged and permutations_left:
        perm = random.choice(permutations_left)
        permutations_left.remove(perm)
        t += 1

        g_score_previous = g_score_previous_origin

        for i in range(num_active_users):
            user_idx = original_ids[perm[i]]
            if user_idx >= num_total_users:
                continue

            subset = perm[:i+1]
            current_g, g_current = cache.get_or_compute(subset, grads_locals, grad_glob_new)  # 使用缓存

            contribution = g_current - g_score_previous
            shapley_values[user_idx] = (t-1)/t * shapley_values[user_idx] + (1/t) * contribution
            g_score_previous = g_current

            if math.fabs(contribution) < args.Tolerance:  # 轮内截断
                break

        # 收敛条件判断（每迭代1次执行）
        shapley_history.append(shapley_values.copy())
        if len(shapley_history) > 10:
            shapley_history.pop(0)  # 保持最近10次记录

        if len(shapley_history) == 10:
            # 计算相对变化率（公式10）
            delta_sum = 0
            for i in range(num_total_users):
                current_val = shapley_history[-1][i]
                if abs(current_val) > 1e-6:  # 避免除零
                    delta_sum += np.mean([abs(shapley_history[-1][i] - shapley_history[k][i]) / abs(current_val)
                                          for k in range(9)])

            avg_delta = delta_sum / num_total_users
            converged = (avg_delta < 0.05)  # 阈值条件

    # softmax方案
    # sum_shapley = 0
    # for idx in idxs_users:
    #     shapley_values[idx] = np.exp(shapley_values[idx])
    #     sum_shapley += shapley_values[idx]
    # for idx in idxs_users:
    #     shapley_values[idx] /= sum_shapley

    shapley_values = signed_normalize(shapley_values, idxs_users)

    return shapley_values

def GMC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)
    original_ids = idxs_users
    cache = ShapleyCache()  # 初始化DP缓存

    if num_active_users == 0 or len(grads_locals) != num_active_users:
        return shapley_values

    try:
        permutations = generate_permutations(num_active_users, args)  # 引导生成排列
    except MemoryError:
        print("Permutations memory error!")
        return shapley_values

    t = 0
    current_g = copy.deepcopy(grad_glob)
    g_score_previous_origin = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0

    for perm in permutations:
        t += 1
        g_score_previous = g_score_previous_origin

        for i in range(num_active_users):
            user_idx = original_ids[perm[i]]
            if user_idx >= num_total_users:
                continue

            subset = perm[:i+1]
            current_g, g_current = cache.get_or_compute(subset, grads_locals, grad_glob_new)  # 使用缓存

            contribution = g_current - g_score_previous
            shapley_values[user_idx] = (t-1)/t * shapley_values[user_idx] + (1/t) * contribution
            g_score_previous = g_current

    # softmax方案
    # sum_shapley = 0
    # for idx in idxs_users:
    #     shapley_values[idx] = np.exp(shapley_values[idx])
    #     sum_shapley += shapley_values[idx]
    # for idx in idxs_users:
    #     shapley_values[idx] /= sum_shapley
    shapley_values = signed_normalize(shapley_values, idxs_users)

    return shapley_values

def GTMC_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)
    original_ids = idxs_users
    cache = ShapleyCache()  # 初始化DP缓存
    if num_active_users == 0 or len(grads_locals) != num_active_users:
        return shapley_values

    try:
        permutations_left = generate_permutations(num_active_users, args) # 引导生成排列
    except MemoryError:
        print("Permutations memory error!")
        return shapley_values

    t, converged = 0, False
    shapley_history = []
    current_g = copy.deepcopy(grad_glob)
    g_score_previous_origin = calculate_dict_cosine_similarity(current_g, grad_glob_new) if current_g else 0

    while not converged and permutations_left:
        perm = random.choice(permutations_left)
        permutations_left.remove(perm)
        t += 1

        g_score_previous = g_score_previous_origin

        for i in range(num_active_users):
            user_idx = original_ids[perm[i]]
            if user_idx >= num_total_users:
                continue

            subset = perm[:i+1]
            current_g, g_current = cache.get_or_compute(subset, grads_locals, grad_glob_new)  # 使用缓存

            contribution = g_current - g_score_previous
            shapley_values[user_idx] = (t-1)/t * shapley_values[user_idx] + (1/t) * contribution
            g_score_previous = g_current

            if math.fabs(contribution) < args.Tolerance:  # 轮内截断
                break

        # 收敛条件判断（每迭代1次执行）
        shapley_history.append(shapley_values.copy())
        if len(shapley_history) > 10:
            shapley_history.pop(0)  # 保持最近10次记录

        if len(shapley_history) == 10:
            # 计算相对变化率（公式10）
            delta_sum = 0
            for i in range(num_total_users):
                current_val = shapley_history[-1][i]
                if abs(current_val) > 1e-6:  # 避免除零
                    delta_sum += np.mean([abs(shapley_history[-1][i] - shapley_history[k][i]) / abs(current_val)
                                          for k in range(9)])

            avg_delta = delta_sum / num_total_users
            converged = (avg_delta < 0.05)  # 阈值条件

    # softmax方案
    # sum_shapley = 0
    # for idx in idxs_users:
    #     shapley_values[idx] = np.exp(shapley_values[idx])
    #     sum_shapley += shapley_values[idx]
    # for idx in idxs_users:
    #     shapley_values[idx] /= sum_shapley
    shapley_values = signed_normalize(shapley_values, idxs_users)

    return shapley_values

def Random_permuation(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    """随机给出贡献"""
    num_total_users = args.num_users
    C_values = np.zeros(num_total_users)
    num_active_users = len(idxs_users)

    # 空值保护
    if num_active_users < 1:
        return C_values

    original_ids = idxs_users  # 原始用户ID列表
    for i in range(num_active_users):
        user_id = original_ids[i]  # 当前用户的原始ID
        C_values[user_id] = random.uniform(-1, 1)

    sum_C = 0
    for idx in idxs_users:
        C_values[idx] = np.exp(C_values[idx])
        sum_C += C_values[idx]
    for idx in idxs_users:
        C_values[idx] /= sum_C
    return C_values


# 定义一部字典，将方法名映射到相应的函数
method_mapping = {
    1: True_Shapley,
    2: MC_Shapley,
    3: TMC_Shapley,
    4: GMC_Shapley,
    5: GTMC_Shapley,
    6: Random_permuation,
}
