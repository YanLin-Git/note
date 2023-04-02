# KL散度、交叉熵
> 参考链接: https://zhuanlan.zhihu.com/p/74075915

## 1. 熵
假设离散随机变量X的概率分布为P(x)，则熵的计算为:  
$$H(P) = - \sum\limits_x P(x) \log P(x)$$

## 2. 相对熵(KL散度)
假设随机变量X有两个单独的概率分布P(x)、Q(x)  
我们可以用KL散度来衡量 这两个分布的差异:  
$$D_{KL}(p||q) = \sum\limits_x p(x) \log \frac {p(x)} {q(x)}$$

> 在机器学习中，P往往用来表示真实分布，Q用来表示模型预测的分布  
> 从上面的计算公式可以看出，Q分布越拟合P分布，**KL散度**越小

## 3. 交叉熵
我们将上面**KL散度**的公式变形:  
$$
\begin{aligned}
    D_{KL}(p||q) &= \sum\limits_x p(x) \log \frac {p(x)} {q(x)} \\
    &= \sum\limits_x p(x) \log p(x) - \sum\limits_x p(x) \log q(x) \\
    &= -H(P) - \sum\limits_x p(x) \log q(x) \quad (1)
\end{aligned}
$$
> 上面(1)式中，第一项 $-H(P)$，即概率分布P的熵  
> 第二项$- \sum\limits_x p(x) \log q(x)$，即**交叉熵**

## 总结
首先，我们希望**损失函数**可以衡量`真实分布`与`预测分布`之间的差异  
应该选择**KL散度**  
但是，真实分布是已知不变的，也就是说$-H(P)$为常数，所以只需要计算第二项，即**交叉熵**