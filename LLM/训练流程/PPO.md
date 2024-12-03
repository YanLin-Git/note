# PPO

> 参考代码: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py  
> 参考博客1: https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html  
> 参考博客2: https://blog.csdn.net/v_JULY_v/article/details/128579457  
> 参考博客3: https://blog.csdn.net/v_JULY_v/article/details/128965854  

## 一、Policy Gradient
1. 给定一个策略 $\pi_\theta$，我们可以使用 $\pi_\theta$ 来生成一系列 $\tau^1、\tau^2、...、\tau^N$，然后计算 $\pi_\theta$ 所能获得的期望奖励
    $$
    \bar{R}_\theta = E_{\tau \sim \pi_\theta} [R(\tau)] = \sum\limits_{\tau} R(\tau) p_\theta (\tau)
    $$
    - $R(\tau)$ 可以由奖励模型给出
    - $ p_\theta (\tau) = p(s_1)p_\theta(a_1|s_1)p(s_2|s_1,a_1)p_\theta(a_2|s_2)p(s_3|s_2,a_2)... = p(s_1) \prod\limits_{t=1}^T p_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t) $
2. 我们的优化目标，就是 $\max\limits_\theta \bar{R}_\theta$
3. 于是对$\bar{R}_\theta$求导，便可得到(1)式:
    $$
    \nabla \bar{R}_\theta = \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a^n_t|s^n_t) \qquad (1)
    $$
    <details>
    <summary>详细求导过程</summary>

    $$
    \begin{aligned}
    \nabla \bar{R}_\theta & = \sum\limits_{\tau} R(\tau) \nabla p_\theta (\tau) & \\
    & = \sum\limits_{\tau} R(\tau) p_\theta (\tau) \frac {\nabla p_\theta (\tau)} {p_\theta (\tau)} & \\
    & = \sum\limits_{\tau} R(\tau) p_\theta (\tau) \nabla \log {p_\theta (\tau)} & 这里用了公式: \frac {\nabla f(x)} {f(x)} = \nabla \log f(x) \\
    & = E_{\tau \sim \pi_\theta} [R(\tau) \nabla \log {p_\theta (\tau)}] & \\
    & = \frac 1 N \sum\limits_{n=1}^N R(\tau^n) \nabla \log {p_\theta (\tau^n)} & \\
    & = \frac 1 N \sum\limits_{n=1}^N R(\tau^n) \nabla \log \left[ p(s^n_1) \prod\limits_{t=1}^{T_n} p_\theta(a^n_t|s^n_t)p(s^n_{t+1}|s^n_t, a^n_t) \right] &  代入p_\theta (\tau) \\
    & = \frac 1 N \sum\limits_{n=1}^N R(\tau^n) \nabla_\theta \left[ \sum\limits_{t=1}^{T_n} \log p_\theta(a^n_t|s^n_t) \right] & 仅保留与\theta有关的项 \\
    & = \frac 1 N \sum\limits_{n=1}^N R(\tau^n) \left[ \sum\limits_{t=1}^{T_n} \nabla \log p_\theta(a^n_t|s^n_t) \right] & 转换为 先求导再求和 \\
    & = \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a^n_t|s^n_t) & (1)
    \end{aligned}
    $$

    </details>

4. PG的完整过程如下:
    ![PolicyGradient.png](../jpgs/PolicyGradient.png)

## 二、PPO

### 2.1、$\nabla \bar{R}_\theta$的改进
- 接下来在上面(1)式的基础上，一步步来改进 $\nabla \bar{R}_\theta$

#### 1) 添加一个baseline
$$
\begin{aligned}
\nabla \bar{R}_\theta & = \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a^n_t|s^n_t) & (1) \\
& \approx \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} [R(\tau^n)-b] \nabla \log p_\theta(a^n_t|s^n_t) & 其中，b \approx E[R(\tau^n)]
\end{aligned}
$$
#### 2) 每个时间步，都拥有一个单独的 $r_t$
- 每一个轨迹 $\tau$，其实是由多个时间步组成的。目前我们对于整个轨迹 $\tau$，都使用一个奖励值 $R(\tau)$
- 其实可以更细致地拆解，每个时刻t，使用不同的奖励值$R_t(\tau)$
    |$s_1$|$s_2$|...|$s_t$|...|$s_T$|在每个时间步t，使用的奖励值$R_t(\tau)$|
    |---|---|---|---|---|---|---|
    |未指定|未指定|未指定|未指定|未指定|$R(\tau)$|$R(\tau)$|
    |$r_1$|$r_2$|...|$r_t$|...|$r_T$|从t时刻到最终T时刻，$r_{t^{\prime}}$ 的一个加权和|
- 具体计算公式为: $R_t(\tau) = \sum\limits_{t^{\prime}=t}^{T} \gamma^{t^{\prime}-t} r_{t^{\prime}}$
- 于是将(1)式进一步改写:
    $$
    \begin{aligned}
    \nabla \bar{R}_\theta & = \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a^n_t|s^n_t) & (1) \\
    & \approx \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} [R_t(\tau^n)-b_t] \nabla \log p_\theta(a^n_t|s^n_t) & 其中，R_t(\tau) = \sum\limits_{t^{\prime}=t}^{T} \gamma^{t^{\prime}-t} r_{t^{\prime}}, b_t \approx E[R_t(\tau^n)] \\
    \end{aligned}
    $$
#### 3)使用优势函数
- 分别来看看$[R_t(\tau^n)-b_t]$中的两项
    1. $R_t(\tau^n) = \sum\limits_{t^{\prime}=t}^{T} \gamma^{t^{\prime}-t} r_{t^{\prime}}$，刻画了从t时刻到最终T时刻，$r_{t^{\prime}}$的一个加权和---->可以用 $Q_{\pi_\theta}(s_t, a_t)$ 来近似替代
    2. $b_t \approx E[R_t(\tau^n)]$，刻画了t时刻的平均奖励值---->可以用 $V_{\pi_\theta}(s_t)$ 来近似替代
- 于是将(1)式进一步改写:
    $$
    \begin{aligned}
    \nabla \bar{R}_\theta & = \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a^n_t|s^n_t) & (1) \\
    & \approx \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} [Q_{\pi_\theta}(s_t, a_t)-V_{\pi_\theta}(s_t)] \nabla \log p_\theta(a^n_t|s^n_t) & \\
    & = \frac 1 N \sum\limits_{n=1}^N \sum\limits_{t=1}^{T_n} A^{\theta}(s_t, a_t) \nabla \log p_\theta(a^n_t|s^n_t) & (2)
    \end{aligned}
    $$
> (2)式中 $A^{\theta}(s_t, a_t)$即为优势函数，再来思考一下它的意义
>    1. $Q_{\pi_\theta}(s, a)$ 刻画了在状态s时，采取动作a后的期望奖励值
>    2. $V_{\pi_\theta}(s_t)$ 刻画了在状态s时的期望奖励值
>    3. 那么它们的差值，就可以用来评估 在状态s时，具体的某个动作a的好坏
>        - $A^{\theta}(s, a) > 0$，采取动作a后，得到的回报比平均值要高，需要增加动作a的概率；反之则降低动作a的概率

### 2.2、重要性采样
- 上面(2)式中 $A^{\theta}(s_t, a_t)$ 是在策略 $\pi_\theta$ 下计算出来的
- 接下来基于重要性采样的原则，从策略 $\pi_\theta$ 换到 策略 $\pi_{\theta^{\prime}}$，对(2)式再做一些变换
    1. 要优化(2)式，等价于优化下式:
        $$
        E_{(s_t,a_t) \sim \pi_\theta} [A^{\theta}(s_t, a_t) \nabla \log p_\theta(a_t|s_t)]
        $$
    2. 将 $A^{\theta}(s_t, a_t)$ 变换成  $A^{\theta^{\prime}}(s_t, a_t)$
        $$
        \begin{aligned}
        & E_{(s_t,a_t) \sim \pi_\theta} \left[ A^{\theta}(s_t, a_t) \nabla \log p_\theta(a_t|s_t) \right] & \\
        = & E_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac {p_\theta(s_t,a_t)} {p_{\theta^\prime}(s_t,a_t)} A^{\theta^\prime}(s_t, a_t) \nabla \log p_\theta(a_t|s_t) \right] & \\
        = & E_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac {p_\theta(a_t | s_t)} {p_{\theta^\prime}(a_t | s_t)} \frac {p_\theta(s_t)} {p_{\theta^\prime}(s_t)} A^{\theta^\prime}(s_t, a_t) \nabla \log p_\theta(a_t|s_t) \right] & \\
        \approx & E_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac {p_\theta(a_t | s_t)} {p_{\theta^\prime}(a_t | s_t)} A^{\theta^\prime}(s_t, a_t) \nabla \log p_\theta(a_t|s_t) \right] & (3) \\
        \end{aligned}
        $$
### 2.3、反推出`目标函数`
- 实际更新参数的时候，我们按照上面的(3)式来更新，于是我们就可以从梯度反推**目标函数**
    $$
    \begin{aligned}
    \nabla_\theta & = E_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac {p_\theta(a_t | s_t)} {p_{\theta^\prime}(a_t | s_t)} A^{\theta^\prime}(s_t, a_t) \nabla \log p_\theta(a_t|s_t) \right] & (3) \\
    & = E_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac {\nabla p_\theta(a_t | s_t)} {p_{\theta^\prime}(a_t | s_t)} A^{\theta^\prime}(s_t, a_t) \right] & 这里用了公式: \frac {\nabla f(x)} {f(x)} = \nabla \log f(x) \\
    \Longrightarrow J(\theta) & = E_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac { p_\theta(a_t | s_t)} {p_{\theta^\prime}(a_t | s_t)} A^{\theta^\prime}(s_t, a_t) \right] & 大功告成！！\\
    \end{aligned}
    $$