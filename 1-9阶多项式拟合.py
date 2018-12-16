"""
高次插值因舍入误差而产生的runge现象
2010年的估计值为249.7,很不理想
"""

import numpy as np
import matplotlib.pyplot as plt

x_array = np.arange(1994, 2004, 1)
y_array = np.array([67.052, 68.008, 69.803, 72.024, 73.400, 72.063, 74.669, 74.487, 74.065, 76.777])
m = 10


def l(i, x):
    temp0 = 1.0
    temp1 = 1.0
    for k in range(m):
        if k != i:
            temp0 *= (x - x_array[k])
            temp1 *= (x_array[i] - x_array[k])
    return temp0/temp1


def p(x):
    temp = 0.0
    for i in range(m):
        temp += y_array[i]*l(i, x)
    return temp


z1 = np.polyfit(x_array, y_array, 9)
x = np.linspace(1993.5, 2003.5, 100000)
y_val = p(x)
plot1 = plt.plot(x_array, y_array, '*', label='original values')
plot2 = plt.plot(x, y_val, "r")

# 预估2010年的石油量
print(p(2010))
plt.savefig("1-9阶多项式拟合.png")
