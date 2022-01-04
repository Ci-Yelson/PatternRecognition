参考PPT上决策面公式：
$$
\begin{array}{c}
g_{i}(X)=X^{T} W_{i} X+W_{i 1}^{T} X+W_{i 0} \\
W_{i}=-\frac{1}{2} \sum_{i}^{-1} \\
W_{i 1}=\sum_{i}^{-1} \mu_{i} \\
W_{i 0}=-\frac{1}{2} \mu_{i}^{T} \sum_{i}^{-1} \mu_{i}-\frac{1}{2} \ln \left|\sum_{i}^{}\right|+\ln P\left(\omega_{i}\right)
\end{array}
$$

$$
决策面方程:

X^{T}\left(W_{i}-W_{j}\right) X+\left(W_{i 1}-W_{j}\right)^{T} X+W_{i 0}-W_{j 0}=0
$$

决策面绘制的一种方法：假设X为D维向量，则可先预先采样X的D-1维坐标，并将其代入决策面方程，代入后该方程就变为了对于最后一维的二次方程，使用求根公式解出最后一维坐标，然后绘制即可。

三维计算过程：
$$
对于X向量：X=\begin{bmatrix}
x&y&z
\end{bmatrix}
\\
假设已知(x,y)，未知z，将决策面方程转换为关于z的二次方程，从而解出z。
$$

$$
\begin{bmatrix}
x&y&z
\end{bmatrix}
\begin{bmatrix}
w_{1,00}&w_{1,01}&w_{1,02}\\
w_{1,10}&w_{1,11}&w_{1,12}\\
w_{1,20}&w_{1,21}&w_{1,22}
\end{bmatrix}
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
+
\begin{bmatrix}
w_{2,0}&w_{2,1}&w_{2,2}
\end{bmatrix}
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
+
w_3=0
$$

$$
P_1 \Rarr x(xw_{1,00}+yw_{1,10}+zw_{1,20})+y(xw_{1,01}+yw_{1,11}+zw_{1,21})+z(xw_{1,02}+yw_{1,12}+zw_{1,22})\\
P_2 \Rarr xw_{2,0}+yw_{2,1}+zw_{2,2}\\
P_3 \Rarr w_3\\
P_1+P_2+P_3=0
$$

$$
w_{1,00}x^2+w_{1,11}y^2+w_{1,21}z^2+(w_{1,10}+w_{1,01})xy+(w_{1,20}+w_{1,02})xz+(w_{1,21}+w_{1,12})yz+w_{2,0}x+w_{2,1}y+w_{2,2}z+w3=0\\
$$

$$
\begin{cases}
az^2+bz+c=0\\
a=w_{1,21}\\
b=(w_{1,20}+w_{1,02})x+(w_{1,21}+w_{1,12})y+w_{2,2}\\
c=w_{1,00}x^2+w_{1,11}y^2+(w_{1,10}+w_{1,01})xy+w_{2,0}x+w_{2,1}y+w_3
\end{cases}
$$

$$
z=\frac{-b\pm\sqrt{b^2-4ac}}{2a}
$$

> 二维也可通过这种方法计算。



参考代码：

```python
# 参数估计
u1 = ...
u2 = ...
sig1 = ...
sig2 = ...

W1 = -0.5 * np.linalg.inv(sig1)
W2 = -0.5 * np.linalg.inv(sig2)
w1 = W1 - W2

W1_1 = np.linalg.inv(sig1) @ u1
W2_1 = np.linalg.inv(sig2) @ u2
w2 = W1_1 - W2_1

W1_0 = -0.5*(u1.T @ np.linalg.inv(sig1) @ u1) - 0.5*np.log(np.linalg.det(sig1)) + np.log(pw1)
W2_0 = -0.5*(u2.T @ np.linalg.inv(sig2) @ u2) - 0.5*np.log(np.linalg.det(sig2)) + np.log(pw2)
w3 = W1_0 - W2_0

x1 = np.arange(150,190,0.01)
x2 = np.arange(40,80,0.01)
x1,x2 = np.meshgrid(x1,x2)

a = w1[2,2]
b = ((w1[0,2]+w1[2,0])*x1 + (w1[1,2]+w1[2,1])*x2 + w2[2])
c = (w1[0,0]*x1*x1 + w1[1,1]*x2*x2 + (w1[0,1]+w1[1,0])*x1*x2 + w2[0]*x1 + w2[1]*x2) + w3

x3_1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
x3_2 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

ax = plt.subplot(121,projection='3d')
# x_te1:男生测试集数据
# x_te2:女生测试集数据
ax.scatter(x_te1[:,0],x_te1[:,1],x_te1[:,2],color='r')
ax.scatter(x_te2[:,0],x_te2[:,1],x_te2[:,2],color='b')
ax.plot_surface(x1,x2,x3_1,color='y')
ax.plot_surface(x1,x2,x3_2,color='y')
```



