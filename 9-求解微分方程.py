"""
y'=y(cost)
y(0)=1
"""
import math
import matplotlib.pyplot as plt
import numpy as np


# 定义y的导数
def y_derivatives(y, t):
    return y*math.cos(t)


# euler法求微分方程
def euler():
    h = [0.1, 0.01, 0.001]
    for h1 in h:
        N = 1/h1
        u = [0 for _ in range(int(N+1))]
        t = [0 for _ in range(int(N+1))]
        u[0] = math.cos(t[0])*1
        for i in range(0, int(N)):
            t[i] = i*h1
            u[i+1] = u[i] + h1*y_derivatives(u[i], t[i])
        plt.scatter(1, u[int(N)], color="red", label="euler")
        print("当步长设置为: " + str(h1) + " Euler法的结果如下")
        print(u[int(N)])


# 改进的euler方法求解微分方程
def euler_plus():
    h = [0.1, 0.01, 0.001]
    for h1 in h:
        N = 1 / h1
        u = [0 for _ in range(int(N + 1))]
        t = [0 for _ in range(int(N + 1))]
        u[0] = 1

        for i in range(0, int(N+1)):
            t[i] = i*h1

        for i in range(0, int(N)):
            u_temp = u[i] + h1*y_derivatives(u[i], t[i])
            u[i+1] = u[i] + (h1*0.5)*(y_derivatives(u[i], t[i])+y_derivatives(u_temp, t[i+1]))

        plt.scatter(1, u[int(N)], color="orange", label="euler_plus")
        print("当步长设置为: " + str(h1) + " 改进的Euler法的结果如下")
        print(u[int(N)])


# 四阶runge_kutta法
def runge_kutta():
    h = [0.1, 0.01, 0.001]
    for h1 in h:
        N = int(1 / h1)
        u = [0 for _ in range(int(N + 1))]
        t = [0 for _ in range(int(N + 1))]
        u[0] = 1
        for i in range(N):
            t[i] = i * h1
            k1 = y_derivatives(u[i], t[i])
            k2 = y_derivatives(u[i]+0.5*h1*k1, t[i]+0.5*h1)
            k3 = y_derivatives(u[i]+0.5*h1*k2, t[i]+0.5*h1)
            k4 = y_derivatives(u[i]+h1*k3, t[i]+h1)
            u[i+1] = u[i]+(h1/6)*(k1+2*k2+2*k3+k4)

        print("当步长设置为: " + str(h1) + " runge_kutta法的结果如下")
        print(u[int(N)])
        plt.scatter(1, u[int(N)], color="blue", label="runge_kutta")


if __name__ == '__main__':
    plt.figure(figsize=(8, 6))
    euler()
    euler_plus()
    runge_kutta()

    print("精确解： ", math.exp(math.sin(1)))
    # 绘图
    x = np.linspace(0, 1, 1000)
    yy = np.exp(np.sin(x))
    plt.plot(x, yy, color="black", label="y=e^sin(t)")
    plt.legend()
    plt.savefig("9-微分方程近似解.png")
