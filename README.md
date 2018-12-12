# 数值分析作业 #

## 1. 9阶多项式拟合问题 ##
- 用九阶多项式拟合曲线，结果为图1
## 2. 最小二乘法拟合直线，抛物线，三次曲线问题 ##
- 用最小二乘法拟合离散数据，结果如图2
## 3. 牛顿法解方程 ##

                    第 1 次迭代
xk的值= 2.6205607476635517

ei = |xi - x*|= 1.6205607476635517

                    第 2 次迭代
xk的值= 1.7084401902625614

ei = |xi - x*|= 0.7084401902625614

ei / (last_ei)^2 的值= 0.2697568987412368

                    第 3 次迭代
xk的值= 1.2063786980181086

ei = |xi - x*|= 0.2063786980181086

ei / (last_ei)^2 的值= 0.4112050941909951

                    第 4 次迭代
xk的值= 1.024161664125103

ei = |xi - x*|= 0.02416166412510301

ei / (last_ei)^2 的值= 0.5672795217855637

                    第 5 次迭代
xk的值= 1.0003814911210203

ei = |xi - x*|= 0.00038149112102026095

ei / (last_ei)^2 的值= 0.6534776653306854

                    第 6 次迭代
xk的值= 1.0000000969928142

ei = |xi - x*|= 9.699281422470563e-08

ei / (last_ei)^2 的值= 0.6664547866875559

                    第 7 次迭代
xk的值= 1.0000000000000062

ei = |xi - x*|= 6.217248937900877e-15

ei / (last_ei)^2 的值= 0.6608747146171305

f(x∗)的二阶导/2(f(x∗)的导数)= 0.6666666666666666

结论：
- 从ei / (last_ei)^2的值可以看出，方程是2阶收敛的。
- 从f(x∗)的二阶导/2(f(x∗)的导数)= 0.6666666666666666可以看出，ei / (last_ei)^2是收敛到常数0.666..的。

## 4-有方程组Hx=b, H为n*n阶 Hilbert矩阵，b为n维单位列向量。n=4,8 ##
1. 用Gauss法求解线性方程组:求出精确解[-4, 60, -180, 140]
2. 用Jacobi法求解: 迭代太慢,python报错栈溢出。尝试解简单方程可以解证明算法无误
3. 用Gauss-Seidel法求解: 迭代7000次[-3.99143925, 59.90802849, -179.78427343, 139.86209779]
4. 用共轭梯度法求解: 共迭代了54次求出了精确地解[-4, 60, -180, 140]

## 5-变成计算三次样条S, 满足S(0)=1,S(1)=3,S(2)=3,S(3)=4,S(4)=2,其中边界条件为S''(0)=S'(0)=0 ##

## 6-取不同的初值,并使用弦截法,来计算方程组y=x^3 + 2x^2 + 10x - 100的根 ##
- 结论
1. 当初始值为x0=5,x1=4时,达到精度1*10^(-5),迭代了5次
2. 当初始值为x0=100,x1=99时,达到精度1*10^(-5),迭代了16次
3. 当初始值为x0=-100,x1=50时,达到精度1*10^(-5),迭代了13次
4. 当初始值为x0=100000,x2=200000时,达到精度1*10^(-5),迭代了42次