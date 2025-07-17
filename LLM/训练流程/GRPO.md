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


## 二、代码实现

- 可以看到:  
    1. [DeepSeekMath](https://arxiv.org/abs/2402.03300)中是在token level去计算
    2. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)则是在sequence level去计算

> 截止到2025.07，[trl](https://github.com/huggingface/trl)库中的**GRPOTrainer**是基于[DeepSeekMath](https://arxiv.org/abs/2402.03300)实现的  
> 还找到一份简单易懂的[参考代码](https://github.com/aburkov/theLMbook/blob/main/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.ipynb)，也是[DeepSeekMath](https://arxiv.org/abs/2402.03300)版本的  
> 于是准备尝试实现下[DeepSeek-V3](https://arxiv.org/abs/2412.19437)版本，比较效果

### 2.1 trainer框架

```python
class Trainer:
    def train():
        # pre: 一些准备工作
        self._inner_training_loop(): # 真正开始训练
            train_dataloader = self.get_train_dataloader() # self.accelerator.prepare(DataLoader(train_dataset, batch_size, collate_fn))
            for epoch in range(num_train_epochs):
                epoch_iterator = train_dataloader
                for step, inputs in enumerate(epoch_iterator):
                    # 核心训练代码
                    tr_loss_step = self.training_step(model, inputs):
                        model.train() # 对应model.eval()
                        inputs = self._prepare_inputs(inputs) # inputs做一些预处理
                        loss = self.compute_loss(model, inputs):
                            outputs = model(**inputs) # 前向传播、计算损失
                        self.accelerator.backward(loss) # 反向传播
                    self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm) # 梯度裁剪
                    self.optimizer.step() # 参数更新
                    self.lr_scheduler.step() # 学习率更新
                    model.zero_grad() # 梯度清零

```

### 2.2 GRPOTrainer的主要改动

```python
class GRPOTrainer:
    def train():
        self._inner_training_loop():
            train_dataloader = self.get_train_dataloader()
            for epoch in range(num_train_epochs):
                epoch_iterator = train_dataloader
                for step, inputs in enumerate(epoch_iterator):
                    tr_loss_step = self.training_step(model, inputs):
                        model.train()
                        inputs = self._prepare_inputs(inputs):
                            self._generate_and_score_completions(generation_batch): # 输入q，批量生成o，并计算相应的r、A
                                # 1. 批量生成o 
                                prompt_completion_ids = unwrapped_model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config)
                                # 2. 计算old_model中，每个token的logp
                                old_per_token_logps = self._get_per_token_logps_and_entropies(self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size)["logps"]
                                # 3. 计算ref_model中，每个token的logp
                                ref_per_token_logps = self._get_per_token_logps_and_entropies(self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep)["logps"]
                                # 4. 计算rewards
                                rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
                                # 5. 计算advantages
                                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                                std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                                advantages = rewards - mean_grouped_rewards
                        loss = self.compute_loss(model, inputs):
                            # 计算当前model，每个token的logp
                            per_token_logps = self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep)["logps"]
                            # 接下来按照 DeepSeekMath 中的公式来计算loss
                        self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()
```

### 2.3 实现自己的_compute_loss

```python
def _compute_loss(self, model, inputs):
    # Compute the per-token log probabilities for the model
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    per_token_logps = self._get_per_token_logps_and_entropies(
        model, input_ids, attention_mask, logits_to_keep
    )["logps"]
    per_token_logps = per_token_logps*completion_mask
    per_seq_logps = per_token_logps.sum(dim=-1)

    # Compute the KL divergence between the model and the reference model
    if self.beta != 0.0:
        ref_per_token_logps = inputs["ref_per_token_logps"]*completion_mask
        ref_per_seq_logps = ref_per_token_logps.sum(dim=-1)
        per_seq_kl = (
            torch.exp(ref_per_seq_logps - per_seq_logps) - (ref_per_seq_logps - per_seq_logps) - 1
        )

    # Compute the loss
    advantages = inputs["advantages"]
    # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
    # old_per_token_logps == per_token_logps, so we can skip it's computation
    # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
    old_per_token_logps = (
        per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
    )
    old_per_token_logps = old_per_token_logps*completion_mask
    old_per_seq_logps = old_per_token_logps.sum(dim=-1)

    coef_1 = torch.exp(per_seq_logps - old_per_seq_logps)
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

    # Two-sided clipping
    if self.args.delta is not None:
        coef_1 = torch.clamp(coef_1, max=self.args.delta)

    per_seq_loss1 = coef_1 * advantages
    per_seq_loss2 = coef_2 * advantages
    per_seq_loss = -torch.min(per_seq_loss1, per_seq_loss2)
    if self.beta != 0.0:
        per_seq_loss = per_seq_loss + self.beta * per_seq_kl

    if self.loss_type == "grpo":
        loss = per_seq_loss.mean()
    # elif self.loss_type == "bnpo":
    #     loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    # elif self.loss_type == "dr_grpo":
    #     loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    return loss
```