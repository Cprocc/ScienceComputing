"""
牛顿迭代法求解方程组x^3+x^2+x-3=0 的根，初始值为x0 = -0.7
迭代7步，并于真值 x*=1 比较
"""


x_true = 1


def f(x):
    """
    y=x^3+x^2+x-3
    :param x:
    :return:y
    """
    return x**3 + x**2 + x - 3


def f_derivatives(x):
    """
    :param x:
    :return: y的导数值
    """
    return 3*(x**2) + 2*x + 1


def f_derivatives2(x):
    """
    :param x:
    :return: y的二阶导数
    """
    return 6*x + 2


def newton_iteration(x):
    """
    牛顿迭代法
    :param x:
    :return:
    """
    return x - f(x) / f_derivatives(x)


if __name__ == '__main__':
    xk = -0.7
    ei_s = []
    for i in range(1, 8):
        print(" "*20 + "第 %s 次迭代" % i + " "*20)
        xk = newton_iteration(xk)
        print("xk的值= " + str(xk))
        ei = abs(xk - x_true)
        print("ei = |xi - x*|= " + str(ei))
        ei_s.append(ei)
        if i != 1:
            print("ei / (last_ei)^2 的值= " + str(ei_s[i-1]/(ei_s[i-2])**2))
        print("")

    print("f(x∗)的二阶导/2(f(x∗)的导数)= " + str(f_derivatives2(x_true)/(2*f_derivatives(x_true))))
