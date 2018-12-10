"""
用最小二乘法拟合1中的数据
分别拟合为 1. 一次曲线 2.抛物线 3.三次曲线
计算均方误差
估计2010年产值
均方误差下,哪个拟合最好
"""
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import numpy as np


x_array = np.arange(1994, 2004, 1)
y_array = np.array([67.052, 68.008, 69.803, 72.024, 73.400, 72.063, 74.669, 74.487, 74.065, 76.777])


def aim_func1(p, x):
    """
    一次多项式拟合的方程
    :param p:
    :param x:
    :return:
    """
    a1, a0 = p
    return a1*x + a0


def aim_func2(p, x):
    """
    二次多项式(曲线)拟合的方程
    """
    a2, a1 , a0 = p
    return a2*(x**2) + a1*x + a0


def error1(p, x, y):
    return aim_func1(p, x)-y


def error2(p, x, y):
    return aim_func2(p, x)-y


p1 = [0, 0]
p2 = [0, 0, 0]
Para1 = leastsq(error1, p1, args=(x_array, y_array))
Para2 = leastsq(error2, p2, args=(x_array, y_array))
a11, a10 = Para1[0]

def f1(x):
    return a11*x + a10
mean1 = 0
mean_square_error1 = 1
for i in range(len(x_array)):
    mean1 += (f1(x_array[i]) - y_array[i])**2
    mean_square_error2 = mean1**0.5
print("拟合为直线时的均方差: " + str(mean_square_error1))
print("估计2010年的值为: " + str(f1(2010)))

a22, a21, a20 = Para2[0]


def f2(x):
    return a22*(x**2) + a21*x + a20
mean2 = 0
mean_square_error2 = 1
for i in range(len(x_array)):
    mean2 += (f2(x_array[i]) - y_array[i])**2
    mean_square_error2 = mean2**0.5
print("拟合曲线为抛物线时的均方差: " + str(mean_square_error2))
print("估计2010年的值为: " + str(f2(2010)))


def three_times_fit():
    # 初始化
    x_array = np.arange(4, 14, 1)
    y_array = np.array([67.052, 68.008, 69.803, 72.024, 73.400, 72.063, 74.669, 74.487, 74.065, 76.777])
    m = len(x_array)
    a = np.zeros((4, 4), dtype='float64')
    b = np.zeros((4, 1))

    def f(i, x):
        """
        基函数span{1,x,x^2,x^3}
        """
        if i == 0:
            return 1
        elif i == 1:
            return x
        elif i == 2:
            return x ** 2
        else:
            return x ** 3

    def compute_eq():
        """
        计算法方程组的两个相关矩阵
        并求出稀疏矩阵res
        :return:
        """
        for i in range(4):
            for j in range(4):
                for x in x_array:
                    a[i][j] += f(i, x) * f(j, x)

        for i in range(4):
            for j in range(len(y_array)):
                b[i] += y_array[j] * f(i, x_array[j])

        a_inv = np.linalg.inv(a)
        res = np.dot(a_inv, b)
        return res

    res = compute_eq()

    def fit_y(x_r):
        x = x_r - 1990
        return res[0] + res[1] * x + res[2] * (x ** 2) + res[3] * (x ** 3)

    x = np.linspace(1994, 2006, 100000)
    plt.plot(x, fit_y(x), color="black", label="y=ax^3+bx^2+cx+d")

    def mean_square_error(x_array, y_array, y):
        mean_error = 0
        for i in range(len(x_array)):
            mean_error += (y_array[i] - y(x_array[i]+1990))**2
            return mean_error**0.5

    mean_square_error3 = mean_square_error(x_array, y_array, fit_y)
    print("拟合三次的均方差是: " + str(float(mean_square_error3)))
    fit2010 = fit_y(2010)
    print("估计2010年的值为: " + str(float(fit2010)))


plt.figure(figsize=(8, 6))
three_times_fit()
plt.scatter(x_array, y_array, color="red", label="Sample Point")
x = np.linspace(1994, 2006, 100000)
y2 = a22*(x*x) + a21*x + a20
plt.plot(x, y2, color="orange", label="y=ax^2+bx+c")
y1 = a11*x + a10
plt.plot(x, y1, color="blue", label="ax+b")

plt.legend()
plt.savefig("最小二乘法321.png")
