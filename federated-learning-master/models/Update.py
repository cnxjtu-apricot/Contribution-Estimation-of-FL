#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, train=True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        if train:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
            self.ldr_test = None
        else:
            self.ldr_test = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=False)  # 评估集
            self.ldr_train = None

    def train(self, net):
        net.train()
        # 初始化优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        last_gradients = {}  # 用于存储最后的梯度字典
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()  # 反向传播计算梯度

                # 保存最后一次的梯度，使用字典存储
                for name, param in net.named_parameters():  # 遍历参数及其名称
                    if param.grad is not None:
                        last_gradients[name] = param.grad.clone()

                optimizer.step()  # 更新参数

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

                # 记录每个 epoch 的损失
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # 返回模型的状态字典和最后的损失均值及最后的梯度
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), last_gradients

    def test(self, net):
        net.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                _, predicted = torch.max(log_probs, 1)  # 返回预测的类别
                all_predictions.extend(predicted.cpu().numpy())  # 将预测结果转为 NumPy 数组
                all_labels.extend(labels.cpu().numpy())  # 将真实标签转为 NumPy 数组

        return np.array(all_predictions), np.array(all_labels)  # 返回 NumPy 数组
