# MLE、MAP、贝叶斯估计

- MLE(极大似然估计)
- MAP(最大后验估计)
- 贝叶斯估计
> 一个通俗易懂的例子: https://zhuanlan.zhihu.com/p/72370235

## MLE(最大似然估计)
认为$\theta$是一个具体的数值，我们的任务是找到最优值  
用数学公式表示:  
$$
    argmax \quad p(D|\theta) = argmax \quad \log p(D|\theta)
$$

## MAP(最大后验估计)
认为$\theta$服从某个分布，我们的任务找到最有可能的$\theta$  
用数学公式表示:  
$$
\begin{aligned}
    argmax \quad p(\theta|D) &= argmax \quad \frac {p(\theta) p(D|\theta)} {p(D)} \\
    &= argmax \quad \frac {p(\theta) p(D|\theta)} {与\theta无关的常数} \\
    &= argmax \quad p(\theta) p(D|\theta) & \quad (1) \\
    &= argmax \quad \log p(\theta) + \log p(D|\theta) & \quad (2)
\end{aligned}
$$
> 上面(1)式中  
> $p(\theta)$ 即**先验分布**的概率  
> $p(D|\theta)$ 即**MLE**

## 举例
1. $p(\theta) \sim N(0,\sigma^2)$  
此时(2)式中的 $\log p(\theta)$， 相当于**l2正则**

1. $p(\theta) \sim 拉普拉斯分布$  
此时(2)式中的 $\log p(\theta)$， 相当于**l1正则**

