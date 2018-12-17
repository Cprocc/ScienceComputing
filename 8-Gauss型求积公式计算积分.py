"""
本例采用Gauss-legendre方法构造Gauss积分
Gauss求积分的步骤
1. 求积区间和权函数构造n+1次正交多项式
2. 解出正交多项式的 n+1个零点作为插值节点
3. 求解n次代数精度得到的n+1个线性方程
"""
import math


def fun1(x):
    return x**2/((1-x**2)**0.5)


def fun2(x):
    return math.sin(x)/x


def main(fun, a, b):
    fun = fun
    a = a
    b = b
    g_l_2 = {0.5773502692: 1}
    g_l_3 = {0.7745966692: 0.555555556, 0: 0.8888888889}
    g_l_5 = {0.9061798459: 0.2369268851, 0.5384693101: 0.4786286705, 0: 0.5688888889}

    items = ['g_l_2', 'g_l_3', 'g_l_5']

    for i in range(len(items)):
        gauss_sum = 0.0
        functions = [g_l_2, g_l_3, g_l_5]
        for key, value in functions[i].items():
            gauss_sum += fun(((b - a) * key + a + b) / 2) * value
            if key > 0:
                gauss_sum += fun(((a - b) * key + a + b) / 2) * value
        gauss_sum = gauss_sum * (b - a) / 2
        print(items[i], ":   ", gauss_sum)


if __name__ == '__main__':
    print("计算函数1的结果 ")
    main(fun1, -1, 1)
    print("计算函数2的结果 ")
    main(fun2, 0, math.pi*0.5)
