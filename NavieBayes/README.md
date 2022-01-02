## 最小错误率贝叶斯分类器（Navie Bayes）

> 图表：
>
> - ROC曲线
> - 决策面
>
> 
>
> 对比：
>
> - 不同先验概率下的对比
> - 不同估计方法的对比



#### 注意

不同先验概率下同一分类器的ROC曲线相同，在绘制时由于判别函数采用正态分布形式而出现的计算误差，导致了出现了不同ROC曲线，在使用对数形式后，ROC曲线一致。



### 实现效果

#### 各分类器ROC曲线

![不同先验概率下各分类器的ROC曲线](https://github.com/Ci-Yelson/PatternRecognition/blob/main/NavieBayes/img/不同先验概率下各分类器的ROC曲线.png)



#### 各分类器决策面

MLE_1RD决策面

![MLE_1RD决策面](https://github.com/Ci-Yelson/PatternRecognition/blob/main/NavieBayes/img/MLE_1RD决策面.png)



MLE_2RD决策面：

![MLE_2RD决策面](https://github.com/Ci-Yelson/PatternRecognition/blob/main/NavieBayes/img/MLE_2RD决策面.png)



MLE_3RD决策面：

![MLE_3RD决策面](https://github.com/Ci-Yelson/PatternRecognition/blob/main/NavieBayes/img/MLE_3RD决策面.png)



KDE_1RD决策面：

![KDE_1RD决策面](https://github.com/Ci-Yelson/PatternRecognition/blob/main/NavieBayes/img/KDE_1RD决策面.png)





### 最大似然估计（MLE）

#### 一维

原理——贝叶斯公式：$p(w_i|x)=\frac{p(x|w_i)p(w_i)}{p(x)}$

步骤：

1. 初始化数据——划分训练集、测试集

2. 估计类条件概率密度——正态分布假设下的最大似然估计

   假设$p(x|w_i)$满足正态分布$p(x|w_i)=\frac{1}{\sqrt{2 \pi \sigma}} \exp \left\{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\right\}$.

   最大似然估计参数：
   $$
   \begin{array}{l}
   \hat{\mu}=\frac{1}{N} \sum_{k=1}^{N} x_{k} \\
   \hat{\sigma}^{2}=\frac{1}{N} \sum_{k=1}^{N}\left(x_{k}-\hat{\mu}\right)^{2}
   \end{array}
   $$
   
3. 使用判别函数计算得分

   二分类问题下，判别函数公式为：
   $$
   \begin{array}{l}
   g(x)=P\left(\omega_{1} \mid x\right)-P\left(\omega_{2} \mid x\right) \\
   g(x)=p\left(x \mid \omega_{1}\right) P(\omega)-p\left(x \mid \omega_{2}\right) P\left(\omega_{2}\right) \\
   g(x)=\ln \frac{p\left(x \mid \omega_{1}\right)}{p\left(x \mid \omega_{2}\right)}+\ln \frac{P\left(\omega_{1}\right)}{P\left(\omega_{2}\right)}
   \end{array}
   $$
   
4. 绘制决策面

   决策面方程：$g(x)=0$

   一维情况下，决策面为一条垂直X轴的直线。

   

5. 性能评估——ROC曲线

   ROC曲线绘制方法：定义三个变量FP,TP,pre_score。将所有样本根据判断函数得分【gx = p(x|w1)p(w1)-p(x|w2)p(w2)】降序排序，然后遍历样本，若为正样本则FP+=1，若为负样本则TP+=1；若当前score!=pre_score，则添加要绘制的坐标点（FP/N, TP/P），同时更新pre_score。



#### 多维

区别于一维情况下的步骤：

2. 估计类条件概率密度——正态分布假设下的最大似然估计

   假设$p(x|w_i)$满足多维正态分布: 
   $$
   p(\boldsymbol{x|w_i})=\frac{1}{(2 \pi)^{d / 2}|\boldsymbol{\Sigma}|^{\frac{1}{2}}} \exp \left\{-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right\}
   \\
   $$

   $$
   \begin{aligned}
   &\text { 式中: } \boldsymbol{x}=\left[x_{1}, x_{2}, \cdots, x_{d}\right]^{\mathrm{T}} \text { 是 } d \text { 维列向量; } \\
   &\mu=\left[\mu_{1}, \mu_{2}, \cdots, \mu_{d}\right]^{\mathrm{T}} \text { 是 } d \text { 维均值向量; } \\
   & \boldsymbol{\Sigma} \text { 是 } d \times d \text { 维协方程矩阵, } \boldsymbol{\Sigma}^{-1} \text { 是 } \boldsymbol{\Sigma} \text { 的逆矩阵, }|\boldsymbol{\Sigma}| \text { 是 } \boldsymbol{\Sigma} \text { 的行列式。 } \\
   &\text { 矩阵 }(\boldsymbol{x}-\boldsymbol{\mu})^{\mathrm{T}} \text { 是矩阵 }(\boldsymbol{x}-\boldsymbol{\mu}) \text { 的转置.}
   \end{aligned}
   $$

   最大似然估计参数：
   $$
   \begin{array}{l}
   \hat{\mu}=\frac{1}{N} \sum_{k=1}^{N} x_{k} \\
   \hat{\Sigma}=\frac{1}{N} \sum_{k=1}^{N}\left(x_{k}-\hat{\mu}\right)\left(x_{k}-\hat{\mu}\right)^{\mathrm{T}}
   \end{array}
   $$



 3.  使用判别函数计算得分

     判别函数公式为：
     $$
     \begin{aligned}
     g_{i}(X) &=\frac{d}{2} \ln 2 \pi-\frac{1}{2} \ln \left|\sum_{i}\right| \\
     &-\frac{1}{2}\left(X-\mu_{i}\right)^{T} \sum_{i}^{-1}\left(X-\mu_{i}\right)+\ln P\left(\omega_{i}\right)
     \end{aligned}
     $$

     

  4.  绘制决策面

      $$
      \begin{array}{c}
      g_{i}(X)=X^{T} W_{i} X+W_{i 1}^{T} X+W_{i 0} \\
      W_{i}=\frac{1}{2} \sum_{i}^{-1} \\
      W_{i 1}=\sum_{i}^{-1} \mu_{i} \\
      W_{i 0}=-\frac{1}{2} \mu_{i}^{T} \sum_{i}^{-1} \mu_{i}-\frac{1}{2} \ln \left|\sum_{i}^{-1}\right|+\ln P\left(\omega_{i}\right)
      \end{array}
      $$
      

​		决策面方程：$g(x)=0$，决策面方程为关于X的二次方程

​		绘制决策面的方法：假设X为D维向量，则可先预先采样D-1维X的坐标，并将其代入决策面方程，代入后该方程就变为了对于最后一维的二次方程，使用求根公式解出最后一维坐标，然后绘制即可。





### Parzen窗（KDE）

各种核函数：

![image-20220101010850650](https://github.com/Ci-Yelson/PatternRecognition/blob/main/NavieBayes/img/核函数)

一维高斯核：
$$
\begin{equation}K(x)=\frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{x^{2}}{2}\right), \quad \hat{p}(x)=\frac{1}{N h} \sum_{i=1}^{n} K\left(\frac{x-x_{i}}{h}\right)\end{equation}
$$
h，也叫bindwidth，可采用经验法则、交叉验证等求解。
$$
\begin{equation}
h=\left(\frac{4 \widehat{\sigma}^{5}}{3 n}\right)^{\frac{1}{5}} \approx 1.06 \hat{\sigma} n^{-1 / 5}
\end{equation}
$$
对于输入特征X，计算 x = X - X_tr，将x代入核函数公式便可求得估计类条件概率。
