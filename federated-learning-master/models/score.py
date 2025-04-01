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


# 假设 args 是你的参数对象，并且你希望调用对应的方法
def evaluate(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    # 使用 get() 来安全地获取方法，如果方法不存在，使用一个默认方法
    method = method_mapping.get(args.FV_method)

    # 调用对应的方法
    result = method(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob)
    return result


def True_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)  # 初始化每个用户的夏普利值为0

    num_users = len(idxs_users)
    # 生成所有参与者的全排列
    permutations = list(itertools.permutations(range(num_users)))

    # 遍历每个排列
    for perm in permutations:
        current_w_subset = w_glob  # 当前全局模型，最开始时是上一轮全局模型
        f1_previous = calculate_F1(current_w_subset)
        for i in range(num_users):
            # 当前子集（前i个成员）
            subset = perm[:i + 1]
            w_subset = FedAvg([w_locals[id_user] for id_user in subset])  # 计算当前子集的模型

            # 计算子集的 F1 分数
            f1_subset = calculate_F1(w_subset)

                # 计算新加入成员的表现分数
            contribution_score = f1_subset - f1_previous
            f1_previous = f1_subset
            shapley_values[idxs_users[perm[i]]] += contribution_score  # 累加表现分数

    # 将表现分数标准化（可选）
    shapley_values /= len(permutations)

    return shapley_values

def TMC_Shapley(w_locals, idxs_users, w_glob, grads_locals):
    pass

def GTG_Shapley(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    shapley_values = np.zeros(num_total_users)  # 初始化每个用户的夏普利值为0

    num_users = len(idxs_users)
    # 生成所有参与者的全排列
    permutations = list(itertools.permutations(range(num_users)))

    # 遍历每个排列
    for perm in permutations:
        current_g_subset = grad_glob  # 当前全局模型，最开始时是全局模型
        if len(grad_glob) != 0:
            g_score_previous = calculate_dict_cosine_similarity(current_g_subset,grad_glob_new)
        else:
            g_score_previous = 0
        for i in range(num_users):
            # 当前子集（前i个成员）
            subset = perm[:i + 1]
            current_g_subset = FedAvg([grads_locals[id_user] for id_user in subset])  # 计算当前子集的grad

            # 计算子集的 F1 分数
            g_subset = calculate_dict_cosine_similarity(current_g_subset,grad_glob_new)

            # 计算新加入成员的表现分数
            contribution_score = g_subset - g_score_previous
            g_score_previous = g_subset
            shapley_values[idxs_users[perm[i]]] += contribution_score  # 累加表现分数

    # 将表现分数标准化（可选）
    shapley_values /= len(permutations)

    return shapley_values

def Leave_One_Out(args, w_locals, idxs_users, w_glob, grads_locals, grad_glob_new, grad_glob):
    num_total_users = args.num_users
    C_values = np.zeros(num_total_users)  # 初始化每个用户的夏普利值为0

    num_users = len(idxs_users)
    all_set = list(range(0, num_users))
    all_set_model = FedAvg([w_locals[id_user] for id_user in range(0, num_users)])  # 本轮全局模型，即总的收益
    f1_best = calculate_F1(all_set_model)

    # 轮流从全集中删除一个参与者
    for i in range(num_users):
        all_set = list(range(0, num_users))
        # 当前子集（去除第i个成员）
        all_set.remove(i)
        w_subset = FedAvg([w_locals[id_user] for id_user in all_set])  # 计算当前子集的模型

        # 计算子集的 F1 分数
        f1_subset = calculate_F1(w_subset)

        # 计算新加入成员的表现分数
        contribution_score = f1_best - f1_subset
        C_values[idxs_users[i]] += contribution_score  # 累加表现分数

    return C_values

def Random_permuation(w_locals, idxs_users, w_glob, grads_locals):
    pass


# 定义一部字典，将方法名映射到相应的函数
method_mapping = {
    1: True_Shapley,
    2: TMC_Shapley,
    3: GTG_Shapley,
    4: Leave_One_Out,
    5: Random_permuation,
}

