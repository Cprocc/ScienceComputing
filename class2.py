"""
用最小二乘法拟合1中的数据
分别拟合为 1. 一次曲线 2.抛物线 3.三次曲线
计算均方误差
估计2010年产值
均方误差下,哪个拟合最好
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from class1 import x_array, y_array

m = len(x_array)
n = 1
A = np.ones(m).reshape((m, 1))
for i in range(n):
    A = np.hstack([A, (x_array ** (i + 1)).reshape((m, 1))])
X = solve(np.dot(A.T, A), np.dot(A.T, y_array.T))

plt.show()
