"""
有方程组Hx=b, H为n*n阶 Hilbert矩阵，b为n维单位列向量。n=4,8
1.用Gauss法求解线性方程组
2.用Jacobi法求解
3.用Gauss-Seidel法求解
4.用共轭梯度法求解
"""
import numpy as np
import copy


def gauss(n):
    l = np.zeros((n, n))
    u = np.zeros((n, n))
    a = np.zeros((n, n))
    b = np.ones((n, 1))

    # 初始化系数矩阵A
    for i in range(n):
        for j in range(n):
            a[i][j] = 1/(i+1+j+1-1)

    # 用Doolittle公式求解l，和u的系数
    for j in range(n):
        u[0][j] = a[0][j]
    for j in range(1, n):
        l[j][0] = a[j][0]/u[0][0]
    for d in range(n):
        l[d][d] = 1

    for i in range(1, n):
        for j in range(i, n):
            temp = 0
            for k in range(i):
                temp += l[i][k] * u[k][j]
            u[i][j] = a[i][j] - temp

        for j in range(i+1, n):
            temp = 0
            for k in range(i):
                temp += l[j][k] * u[k][i]
            l[j][i] = (a[j][i]-temp)/u[i][i]

    u_inv = np.linalg.inv(u)
    l_inv = np.linalg.inv(l)
    x = np.dot(u_inv, l_inv)
    x = np.dot(x, b)
    return x


def jacobi(n):
    a = np.zeros((n, n))
    b = np.ones((n, 1))
    # 系数矩阵A初始化
    for i in range(n):
        for j in range(n):
            a[i][j] = 1/(i+1+j+1-1)
    # 初始解x0 = (0,0,0...共n个0)
    x1 = np.array([[0.0], [0.0], [0.0], [0.0]])
    for _ in range(200):
        x = copy.copy(x1)
        for i in range(n):
            temp = 0
            for j in range(n):
                if i != j:
                    temp += a[i][j]*x[j][0]
            x1[i][0] = (b[i][0] - temp)/a[i][i]
    return x1


def gauss_seidel(n):
    a = np.zeros((n, n))
    b = np.ones((n, 1))
    # 系数矩阵A初始化
    for i in range(n):
        for j in range(n):
            a[i][j] = 1/(i+1+j+1-1)
    # 初始解x0 = (0,0,0...共n个0)
    x = np.ones((n, 1))
    for _ in range(7000):
        for i in range(n):
            temp = 0
            for j in range(n):
                temp += a[i][j]*x[j][0]
            x[i][0] = x[i][0] + (b[i][0] - temp)/a[i][i]
    return x


def c_g(n):
    a = np.zeros((n, n))
    b = np.ones((n, 1))
    # 系数矩阵A初始化
    for i in range(n):
        for j in range(n):
            a[i][j] = 1 / (i + 1 + j + 1 - 1)
    # 初始解x0 = (0,0,0...共n个0)
    x = np.ones((n, 1))


if __name__ == '__main__':
    # print(gauss(4))
    # print(gauss(8))
    # jacobi(4)
    # gauss_seidel(4)
    pass


