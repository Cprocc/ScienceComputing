"""
f(x)=e^(3x)cos(Pi*x),考虑f(x)在(0,2Pi)上的积分。
区间等分为50,100,200,500,1000
分别用复合梯形公式以及复合Simpson积分公式计算积分值
与精确值比较,列表说明误差的收敛性
"""
import numpy as np


def f(x):
    return (np.exp(3*x))*(np.cos(np.pi*x))


def composite_trapezoidal(n):
    """
    分成n份的复合梯形公式,区间为(0,2pi)
    :param n:
    :return:
    """
    a = 0
    b = 2*np.pi
    h = (b-a)/n
    temp = 0
    for i in range(1, n):
        temp += f(h*i)
    return ((b-a)/(2*n))*(f(a)+f(b)+2*temp)


def composite_simpson(n):
    a = 0
    b = np.pi*2
    h = (b-a)/n
    temp1 = 0
    temp2 = 0
    for i in range(1, n):
        temp1 += f(h*i)
    for i in range(0, n):
        temp2 += f((0.5+i)*h)
    return ((b-a)/(6*n))*(f(a)+2*temp1+4*temp2+f(b))


if __name__ == '__main__':
    test_n = [50, 100, 200, 500, 1000]
    for i in range(len(test_n)):
        print("取等分区间n= " + str(test_n[i]) + " 时的积分结果")
        print(composite_trapezoidal(test_n[i]))
        print(composite_simpson(test_n[i]))
