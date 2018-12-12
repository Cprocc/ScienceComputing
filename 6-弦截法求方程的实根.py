"""
用不同的初始值,并使用弦截法,来计算方程组y=x^3 + 2x^2 + 10x - 100的根
"""


import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(8, 6))
x = np.linspace(-2, 4, 1000000)
y = x**3 + 2*(x**2) + 10*x - 100
plt.plot(x, y, color="orange", label="y=x^3 + 2x^2 + 10x - 100")
plt.legend()
plt.savefig("6-弦截法求方程的实根-画图判别个数.png")

"""
经画图判别上述方程只有一个实根(且实根在3.5附近)，可用简单的弦截法计算。
"""


def f(x):
    return x**3 + 2*(x**2) + 10*x - 100


def secant_method(x0, x1):
    return x1 - f(x1)/(f(x1) - f(x0))*(x1 - x0)


def re_secant_method(x0, x1):
    time = 0
    while True:
        temp = secant_method(x0, x1)
        x0 = x1
        x1 = temp
        time += 1
        if abs(f(x1)) - 0 < 0.00001:
            print("迭代了 " + str(time) + " 次")
            print(x1)
            break


if __name__ == '__main__':
    print("当初始值选5, 4时 ")
    re_secant_method(5, 4)
    print("当初始值选100, 99时 ")
    re_secant_method(100, 99)
    print("当初始值选取-100, 50时 ")
    re_secant_method(-100, 50)
    print("当初始值选取100000, 200000时 ")
    re_secant_method(100000, 200000)
