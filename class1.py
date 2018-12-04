"""
高次插值因舍入误差而产生的runge现象
2010年的估计值为249.7,很不理想
"""

import numpy as np
import matplotlib.pyplot as plt

x_array = np.arange(1994, 2004, 1)
y_array = np.array([67.052, 68.008, 69.803, 72.024, 73.400, 72.063, 74.669, 74.487, 74.065, 76.777])
n = 9
m = len(x_array)

z1 = np.polyfit(x_array, y_array, 9)
p1 = np.poly1d(z1)
print(p1)
y_val = p1(x_array)
plot1 = plt.plot(x_array, y_array, '*', label='original values')
plot2 = plt.plot(x_array, y_val, "r")

# 预估2010年的石油量
print(p1(2010))
plt.show()


