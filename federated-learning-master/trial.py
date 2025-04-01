import numpy as np
from itertools import permutations
import random

def generate_combinations(n, m):
    # 生成 n 个参与者的索引
    participants = np.arange(n)

    # 取出前 m 位的全排列
    fixed_indices = participants[:m]
    all_permutations = list(permutations(fixed_indices))

    # 找出剩余参与者
    remaining_indices = participants[m:]

    # 结果列表
    result = []

    # 对于每个固定的排列，随机获取剩余的 n - m 位参与者的排列
    for perm in all_permutations:
        # 随机排列剩余参与者
        random_remaining = random.sample(remaining_indices.tolist(), len(remaining_indices))

        # 组合排列
        combined = list(perm) + random_remaining
        result.append(combined)

    return result

# 示例参数
n = 6  # 总参与者数
m = 3  # 需要全排列的参与者数

# 生成组合
combinations = generate_combinations(n, m)

# 打印结果
for index, combination in enumerate(combinations):
    print(f"组合 {index + 1}: {combination}")