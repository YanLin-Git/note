# DPO


## 回顾instructGPT

- [instructGPT](LLM/训练流程/instructGPT.md)中，`偏好数据训练`分为两个阶段
    1. 训练奖励模型`RM`
        - 可以理解为训练一个pair-wise的排序模型，优化目标 $\textcircled{1}$:
        $$
        \max_{\phi} \quad E_{(x,y_w,y_l) \sim D} \{log\ \sigma[r_{\phi}(x, y_w) - r_{\phi}(x, y_l)]\} \qquad (1)
        $$
    2. `RLHF`
        - `RM`训练好之后，固定 $r_{\phi}$，再使用`PPO`算法进行强化学习，优化目标 $\textcircled{2}$:
        $$
        \max_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x,y)] - \beta D_{KL} \{ \pi_{\theta}(y|x) || \pi_{ref}(y|x) \} \qquad (2)
        $$

## DPO的推导过程
- `PPO`算法比较复杂，而且不够稳定。`DPO`简化了整个训练流程，我们来看看推导过程
    > `PPO`中，先去优化目标 $\textcircled{1}$，再去优化目标 $\textcircled{2}$  
    > `DPO`则通过理论推导，将两个目标 $\textcircled{1}\textcircled{2}$ 一起优化，不再去显式地训练一个奖励模型`RM`
### 一、先通过目标 $\textcircled{2}$， 寻找 $r_{\phi}$ 与 $\pi_{\theta}$ 之间的关系
1. 首先对(2)式进一步推导
    $$
    \begin{aligned}
    & \max_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x,y)] - \beta D_{KL} \{ \pi_{\theta}(y|x) || \pi_{ref}(y|x) \} & (2) \\
    = & \max_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x,y)] - \beta \sum\limits_{x \sim D, y \sim \pi_{\theta}(y|x)} \pi_{\theta}(y|x) \log \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x)} & \text{代入KL散度计算公式} \\
    = & \max_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x,y)] - \beta E_{x \sim D, y \sim \pi_{\theta}(y|x)} \log \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x)} & \text{转化为期望} \\
    = & \max_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ r_{\phi}(x,y) - \beta \log \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x)} \right]  & \text{即 instructGPT论文中的公式} \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ \beta \log \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x)} - r_{\phi}(x,y) \right]  & \text{转化为最小化问题} \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ \log \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x)} - \frac 1 \beta r_{\phi}(x,y) \right]  & \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ \log \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}} \right]  & \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \log \left[ \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}} \frac {Z(x)} {Z(x)} \right]  & \text{引入}Z(x) \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ \log  \frac {\pi_{\theta}(y|x)} { \frac 1 {Z(x)} \pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}} - \log Z(x) \right]  & (3) \\
    \end{aligned}
    $$

    > 这里的 $Z(x)$ 可以任意指定，`DPO`中选用下式:
    $$
    Z(x) = \sum\limits_y \pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}
    $$
    > 这样有两个好处:
    >    1. $Z(x)$ 与 $\pi_{\theta}$ 无关
    >    2. 对(3)式中分母作了归一化，于是可将分母视为一种概率分布，记为 $\pi^{\ast}(y|x)$。即:
        $$
        \pi^{\ast}(y|x) = \frac {1} {Z(x)} \pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)} = \frac {\pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}} { \sum\limits_y \pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}}
        $$

2. 于是(3)式可以继续化简:
    $$
    \begin{aligned}
    & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ \log  \frac {\pi_{\theta}(y|x)} { \frac 1 {Z(x)} \pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}} - \log Z(x) \right]  & (3) \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ \log  \frac {\pi_{\theta}(y|x)} {\pi^{\ast}(y|x)} - \log Z(x) \right]  & \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D, y \sim \pi_{\theta}(y|x)} \left[ \log  \frac {\pi_{\theta}(y|x)} {\pi^{\ast}(y|x)} \right] - E_{x \sim D, y \sim \pi_{\theta}(y|x)} \log Z(x) & \\
    = & \min_{\pi_{\theta}} \quad E_{x \sim D} \{ D_{KL} \left[ \pi_{\theta}(y|x) || \pi^{\ast}(y|x) \right] \} - E_{x \sim D} \log Z(x) & (4)\\
    \end{aligned}
    $$
3. 上面的(4)式中，后半部分与 $\pi_\theta$ 无关，只需关注前半部分。由KL散度的知识，我们知道，两个分布完全一致时，KL散度最小。即:
    $$
    \pi_{\theta}(y|x) = \pi^{\ast}(y|x) = \frac {1} {Z(x)} \pi_{ref}(y|x) e^{\frac 1 \beta r_{\phi}(x,y)}
    $$
    变化一下，我们就得到了$r_{\phi}$ 与 $\pi_{\theta}$ 之间的关系:
    $$
    r_{\phi}(x,y) = \beta \log \frac {\pi_{\theta}(y|x)} {\pi_{ref}(y|x)} + \beta \log Z(x) \qquad (5)
    $$
### 二、将推导出来的(5)式，代入到目标 $\textcircled{1}$
$$
\begin{aligned}
& \max_{\phi} \quad E_{(x,y_w,y_l) \sim D} \{log\ \sigma[r_{\phi}(x, y_w) - r_{\phi}(x, y_l)]\} & (1) \\
= & \max_{\theta} \quad E_{(x,y_w,y_l) \sim D} \left\{log\ \sigma \left[ \beta \log \frac {\pi_{\theta}(y_w|x)} {\pi_{ref}(y_w|x)} - \beta \log \frac {\pi_{\theta}(y_l|x)} {\pi_{ref}(y_l|x)} \right] \right\} & (6)
\end{aligned}
$$
- 至此，我们便得到了`DPO`最终的**优化目标**

<details>
<summary><b>代码示例</b></summary>

```python
# todo
```

</details>

## DPO的梯度更新
- todo