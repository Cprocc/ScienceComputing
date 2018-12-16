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
    for _ in range(100000):
        for i in range(n):
            temp = 0
            for j in range(n):
                temp += a[i][j]*x[j][0]
            x[i][0] = x[i][0] + (b[i][0] - temp)/a[i][i]
    return x


def inner_product(a, b):
    temp = 0
    for i in range(len(a)):
        temp += a[i]*b[i]
    return temp


def c_g(n):
    a = np.zeros((n, n))
    b = np.ones((n, 1))
    # 系数矩阵A初始化
    for i in range(n):
        for j in range(n):
            a[i][j] = 1 / (i + 1 + j + 1 - 1)
    # 初始解x0 = (0,0,0...共n个0)
    r0 = np.zeros((n, 1))
    x = np.ones((n, 1))
    r = b - np.dot(a, x)
    p = r
    time = 0
    while True:
        if (r == r0).all():
            print("C-G法迭代了 " + str(time) + " 次 ")
            return x
        else:
            ap = np.dot(a, p)
            if inner_product(p, ap) == 0:
                print("C-G法迭代了 " + str(time) + " 次 ")
                return x
            else:
                alpha = inner_product(r, r)/(inner_product(p, ap))
                x1 = x + alpha*p
                r1 = r - alpha*ap
                beat = inner_product(r1, r1)/inner_product(r, r)
                p1 = r1 + beat*p
        r = copy.copy(r1)
        x = copy.copy(x1)
        p = copy.copy(p1)
        time += 1


if __name__ == '__main__':
    # print(gauss(4))
    # print(gauss(8))
    # # jacobi(4)
    # # jacobi(8)
    # print(gauss_seidel(4))
    print(gauss_seidel(8))
    print(c_g(4))
    print(c_g(8))

