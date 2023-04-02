# ICA(独立成分分析)

先用**鸡尾酒会问题**(cocktail party problem)为例:  
在一个聚会场合中，有 $n$ 个人同时说话，而屋子里的任意一个话筒录制到底都只是叠加在一起的这 $n$ 个人的声音。但如果假设我们也有 $n$ 个不同的话筒安装在屋子里，并且这些话筒与每个说话人的距离都各自不同，那么录下来的也就是不同的组合形式的所有人的声音叠加。使用这样布置的 $n$ 个话筒来录音，能不能区分开原始的 $n$ 个说话者每个人的声音信号呢？

## 0. 预备知识

这个问题用数学公式来表示就是:  
> 假设有某个样本数据 $s \in R^n$，这个数据是由 $n$ 个独立的来源生成的。我们观察到的为x

$$ x = As $$

展开来看，就是这样:  
$$
\begin{aligned}
    \begin{pmatrix}
        x_{11} & x_{12} & \cdots & x_{1m}\\
        x_{21} & x_{22} & \cdots & x_{2m}\\
        \vdots & \vdots & \ddots & \vdots\\
        x_{n1} & x_{n2} & \cdots & x_{nm}
    \end{pmatrix}
    = 
    \begin{pmatrix}
        a_{11} & a_{12} & \cdots & a_{1n}\\
        a_{21} & a_{22} & \cdots & a_{2n}\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n1} & a_{n2} & \cdots & a_{nn}
    \end{pmatrix}
    \begin{pmatrix}
        s_{11} & s_{12} & \cdots & s_{1m}\\
        s_{21} & s_{22} & \cdots & s_{2m}\\
        \vdots & \vdots & \ddots & \vdots\\
        s_{n1} & s_{n2} & \cdots & s_{nm}
    \end{pmatrix}
\end{aligned}
$$

1. $s_{i}$表示第i个说话者的声音，$s_{ij}$表示第i个说话者在j时刻的声音
2. 那么$x_{11}$就表示$(s_{11},s_{21},\cdots,s_{n1})$的一个线性组合，即j时刻，所有说话者声音的组合，也就是我们观察到的第一个话筒录制声音
    - 同理$x_{21}$表示它们的另外一个线性组合
    - ...
3. 于是矩阵A就称为**混合矩阵**，对应的也就有一个**还原矩阵**，记为W:  
$$ x = As \Rightarrow A^{-1}x = s \Rightarrow s=Wx $$

也展开看看:  

$$
\begin{aligned}
    \begin{pmatrix}
        s_{11} & s_{12} & \cdots & s_{1m}\\
        s_{21} & s_{22} & \cdots & s_{2m}\\
        \vdots & \vdots & \ddots & \vdots\\
        s_{n1} & s_{n2} & \cdots & s_{nm}
    \end{pmatrix}
    = 
    \begin{pmatrix}
        w_{11} & w_{12} & \cdots & w_{1n}\\
        w_{21} & w_{22} & \cdots & w_{2n}\\
        \vdots & \vdots & \ddots & \vdots\\
        w_{n1} & w_{n2} & \cdots & w_{nn}
    \end{pmatrix}
    \begin{pmatrix}
        x_{11} & x_{12} & \cdots & x_{1m}\\
        x_{21} & x_{22} & \cdots & x_{2m}\\
        \vdots & \vdots & \ddots & \vdots\\
        x_{n1} & x_{n2} & \cdots & x_{nm}
    \end{pmatrix}
\end{aligned}
$$

我们想要得到某个说话者的声音$s_{i}$，只需要使用W中的第i行$W_i$去乘上矩阵x

## 1. 概率解释

我们假设每个声源的分布 $s_i$ 都是通过密度函数 $p_s$ 给出，然后联合分布 $s$ 则为：

$$
    p(s)=\prod_{i=1}^n p_s(s_i)
$$
> 这里的$p(s)$可以有多种选择，只要不是高斯分布就可以  
> 一种实验效果不错的函数是这个:
$$
    F(s) = sigmoid(s) = g(s) = \frac 1 {1+e^{-s}}\\
    p(s) = F'(s)
$$

这里要注意，通过在建模中将联合分布（joint distribution）拆解为边界分布（marginal）的乘积（product），就能得出每个声源都是独立的假设（assumption）。那么对应的 $x = As = W^{-1}s$ 的密度函数为：

$$
    p(x)=\prod_{i=1}^n p_s(W_i x)\cdot |W|
$$

## 2. 最大似然估计
上面已经构建好了一个参数为W的模型，写出对数似然函数:  
$$
    l(W)=\sum_{i=1}^m(\sum_{j=1}^n log g'(w_j^Tx^{(i)})+log|W|))
$$

## 3. 梯度上升法
于是更新规则为:  
$$
    W:=W+\alpha\begin{pmatrix}
    \begin{bmatrix}
    1-2g(w_1^T x^{(i)}) \\
    1-2g(w_2^T x^{(i)}) \\
    \vdots \\
    1-2g(w_n^T x^{(i)})
    \end{bmatrix}x^{(i)T} + (W^T)^{-1}
    \end{pmatrix}
$$

> 上式中的 $\alpha$ 是学习速率（learning rate）
