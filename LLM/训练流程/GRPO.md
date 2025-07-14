# GRPO

## 一、优化目标
1. [DeepSeekMath](https://arxiv.org/abs/2402.03300)中首次提出，公式如下:
    $$
    \begin{aligned}
    J_{GRPO}(\theta) & = E[q \sim P(Q), {\{o_i\}}^G_{i=1} \sim \pi_{\theta_{old}}(O|q)] \\
    & = \frac 1 G \sum\limits_{i=1}^G \quad \frac 1 {|o_i|} \sum\limits_{t=1}^{|o_i|} \left\{ min \left[ \frac {\pi_\theta(o_{i,t}|q, o_{i,<t})} {\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, clip \left( \frac {\pi_\theta(o_{i,t}|q, o_{i,<t})} {\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})},1-\varepsilon,1+\varepsilon \right) \hat{A}_{i,t} \right] - \beta D_{KL} [\pi_\theta || \pi_{ref}] \right\}
    \end{aligned}
    $$


    <details>
    <summary>KL散度(token level)</summary>

    $$
    D_{KL}[\pi_\theta||\pi_{ref}] = \frac {\pi_{ref}(o_{i,t}|q, o_{i,<t})} {\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - \log \frac {\pi_{ref}(o_{i,t}|q, o_{i,<t})} {\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - 1
    $$
    
    </details>


    <details>
    <summary>每个token的优势函数</summary>
    
    > 整个sequence中的每个token，都使用相同的计算方式:
    
    $$
    \hat{A}_{i,t} = \frac {r_i - mean(\{r_1,r_2,...,r_G\})} {std(\{r_1,r_2,...,r_G\})}
    $$

    </details>
    
2. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)中，做了一些简化:
    $$
    \begin{aligned}
    J_{GRPO}(\theta) & = E[q \sim P(Q), {\{o_i\}}^G_{i=1} \sim \pi_{\theta_{old}}(O|q)] \\
    & = \frac 1 G \sum\limits_{i=1}^G \left\{ min \left[ \frac {\pi_\theta(o_i|q)} {\pi_{\theta_{old}}(o_i|q)} A_i, clip \left( \frac {\pi_\theta(o_i|q)} {\pi_{\theta_{old}}(o_i|q)},1-\varepsilon,1+\varepsilon \right) A_i \right] - \beta D_{KL} [\pi_\theta || \pi_{ref}] \right\}
    \end{aligned}
    $$


    <details>
    <summary>KL散度(sequence level)</summary>

    $$
    D_{KL}[\pi_\theta||\pi_{ref}] = \frac {\pi_{ref}(o_i|q)} {\pi_{\theta}(o_i|q)} - \log \frac {\pi_{ref}(o_i|q)} {\pi_{\theta}(o_i|q)} - 1 \\
    $$

    </details>


    <details>
    <summary>每个sequence的优势函数</summary>

    $$
    A_i = \frac {r_i - mean(\{r_1,r_2,...,r_G\})} {std(\{r_1,r_2,...,r_G\})}
    $$

    </details>


3. 代码实现
    - 可以看到:  
        1. [DeepSeekMath](https://arxiv.org/abs/2402.03300)中是在token level去计算
        2. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)则是在sequence level去计算
    - 找到一份[DeepSeekMath](https://arxiv.org/abs/2402.03300)对应的[参考代码](https://github.com/aburkov/theLMbook/blob/main/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb)
