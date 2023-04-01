# 高斯判别模型(Gaussian Discriminant Analysis)

## 0. 多元高斯分布

$n$维多元正态分布，也叫做多变量高斯分布，参数为一个$n$维 **均值向量** $\mu \in  R^n $，以及一个 **协方差矩阵** $\Sigma \in  R^{n\times n}$，其中$\Sigma \geq 0$ 是一个对称（symmetric）的半正定（positive semi-definite）矩阵。当然也可以写成"$N (\mu, \Sigma)$" 的分布形式，密度（density）函数为：

$$
p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$$

类似一元高斯分布，我们有:  
$$
    E[X] = \int_x xp(x;\mu,\Sigma)dx=\mu \\
    Cov(X) = E[(X-E[X])(X-E[X])^T =\Sigma
$$

> 参考链接: https://www.zhihu.com/question/36339816/answer/385944057

## 1 GDA

### 1.1 概率解释

假如我们有一个分类问题，其中输入特征 $x$ 是一系列的连续随机变量（continuous-valued random variables），那就可以使用高斯判别分析（Gaussian Discriminant Analysis ，缩写为 GDA）模型，其中对 $p(x|y)$用多元正态分布来进行建模。这个模型为：

$$
\begin{aligned}
    y & \sim Bernoulli(\phi)\\
    x|y = 0 & \sim N(\mu_o,\Sigma)\\
    x|y = 1 & \sim N(\mu_1,\Sigma)\\
\end{aligned}
$$

然后根据**贝叶斯规则**来计算:  
$$
    p(y|x) = \frac {p(x|y) p(y)} {\sum_y p(x|y) p(y)}
$$

**分布写出来的具体形式如下**：

$$
\begin{aligned}
    p(y) & =\phi^y (1-\phi)^{1-y}\\
    p(x|y=0) & = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp ( - \frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)  )\\
    p(x|y=1) & = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp ( - \frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)  )\\
\end{aligned}
$$

在上面的等式中，模型的参数包括$\phi, \Sigma, \mu_0 和 \mu_1$。（要注意，虽然这里有两个不同方向的均值向量$\mu_0$ 和 $\mu_1$，针对这个模型还是一般只是用一个协方差矩阵$\Sigma$。）

### 1.2 最大似然估计

$$
\begin{aligned}
l(\phi,\mu_0,\mu_1,\Sigma) &= \log \prod^m_{i=1}p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)\\
&= \log \prod^m_{i=1}p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi)\\
\end{aligned}
$$

通过使 $l$ 取得最大值，找到对应的参数组合，然后就能找到该参数组合对应的最大似然估计，如下所示：

$$
\begin{aligned}
    \phi & = \frac {1}{m} \sum^m_{i=1}1\{y^{(i)}=1\}\\
    \mu_0 & = \frac{\sum^m_{i=1}1\{y^{(i)}=0\}x^{(i)}}{\sum^m_{i=1}1\{y^{(i)}=0\}}\\
    \mu_1 & = \frac{\sum^m_{i=1}1\{y^{(i)}=1\}x^{(i)}}{\sum^m_{i=1}1\{y^{(i)}=1\}}\\
    \Sigma & = \frac{1}{m}\sum^m_{i=1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\\
\end{aligned}
$$

### 1.3 高斯判别分析（GDA）与逻辑回归（logistic regression）

对于二分类问题，如果$p(x|y)$是一个多变量的高斯分布（且具有一个共享的协方差矩阵$\Sigma$），那么$p(y|x)$则必然符合一个逻辑函数（logistic function）
> 也就是说，GDA，相当于逻辑回归

再推广一下，二分类问题中:  
$$
\begin{aligned}
    y & \sim Bernoulli(\phi)\\
    x|y = 0 & \sim Exponential Family(\eta)\\
    x|y = 1 & \sim Exponential Family(\eta)\\
\end{aligned}
$$
只要这里的$p(x|y)$是同一种指数分布，那么此时的模型，就相当于**逻辑回归**  
> 例如 $x|y \sim P(\lambda)、x|y \sim E(\lambda)$等等

**更多**  
课上有学生提出这样一个问题，吴恩达老师回答，`应该是的，可以证明一下`
$$
\begin{aligned}
    y & \sim Multinoulli(\phi_1,\phi_2,\cdots,\phi_k)\\
    x|y = 0 & \sim Exponential Family(\eta)\\
    x|y = 1 & \sim Exponential Family(\eta)\\
    \cdots \\
    x|y = k & \sim Exponential Family(\eta)\\
\end{aligned}
$$
那么此时的模型，就相当于**softmax**