import numpy as np  
import matplotlib.pyplot as plt  

# 逐步上升
def step_increase(t, initial_value=1):
    return initial_value * t

# 指数上升
def Exp_increase(t, initial_value=1, sti_rate = 6):
    return initial_value * np.exp(sti_rate * (t + 1)/ T)

# 多项式上升
def polynomial_increase(t, initial_value=1, p=2):
    return initial_value * t ** p

# 设置总时间步长
T = 20

# 计算上升的值
step_values = [step_increase(t) for t in range(1, T + 1)]
exp_values = [Exp_increase(t) for t in range(1, T + 1)]
poly_values = [polynomial_increase(t) for t in range(1, T + 1)]

# 绘制结果
t_values = np.arange(1, T + 1)

plt.figure(figsize=(12, 6))
plt.plot(t_values, step_values, label='Step Increase', marker='o')
plt.plot(t_values, exp_values, label='Exp Increase(stimulate = 6)', marker='s')
plt.plot(t_values, poly_values, label='Polynomial Increase(p = 2)', marker='D')

plt.xlabel('epoch', fontsize=20)
plt.ylabel('g_diff', fontsize=20)
plt.xticks(t_values)
plt.legend()

plt.savefig(f'./save/epoch_g_diff_comparison.png',
            dpi=300,
            bbox_inches='tight')