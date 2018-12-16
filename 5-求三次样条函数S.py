import matplotlib.pyplot as plt
import numpy as np
from sympy import *


# 初始化矩阵, n为阶数, Am=g即为要求的最终形式, lambda1, miu为中间变量
n = 4
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 3, 4, 2])
A = np.zeros((5, 5))
lambda1 = np.zeros(5)
miu = np.zeros(5)
h = np.zeros(5)
g = np.zeros(5)

# 根据课本180/181 页公式初始化g, A
for i in range(1, n):
    lambda1[i] = 0.5
    miu[i] = 0.5

for i in range(n):
    h[i] = 1

g[0] = 6
g[4] = -6
for i in range(1, n):
    g[i] = 3*(miu[i]*(y[i+1]-y[i])/h[i] + lambda1[i]*(y[i]-y[i-1])/h[i-1])

A[0][0] = 2
A[0][1] = 1
A[4][3] = 1
A[4][4] = 2
for i in range(1, n):
    A[i][i] = 2
    A[i][i-1] = lambda1[i]
    A[i][i+1] = miu[i]

# 解出m
A_inv = np.linalg.inv(A)
m = np.dot(A_inv, g)

# 确定n个区间内的函数
xx = Symbol('x')
for i in range(n):
    s = ((h[i]+2*(xx - x[i+1]))/(h[i]**3))*((xx - x[i+1])**2)*y[i] + \
        ((h[i]-2*(xx - x[i+1]))/(h[i]**3))*((xx - x[i])**2)*y[i+1] + \
        (xx - x[i])*((xx-x[i+1])**2)*m[i]/(h[i]**2) + (xx - x[i+1])*((xx-x[i])**2)*m[i+1]/(h[i]**2)
    print(s)

