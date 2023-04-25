# word2vec

## 1. skip-gram(跳字模型)
### 1.1 概率解释
> 根据中心词预测上下文
1. 给定中心词，上下文中某个词出现的概率为:

$$
    P(w_o|w_c) = \frac {exp( u_o^T v_c)} {\sum\limits_i exp(u_i^T v_c)}
$$

2. 给定中心词，指定上下文窗口为m，此时的条件概率为:

$$
    \prod\limits_{-m \le j \le m, j \ne 0} P(w_{c+j}|w_c)
$$

3. 给定一个长度为T的文本序列，设时间步t的词为 $w_t$ ，此时的条件概率为:

$$
    \prod\limits_{t=1}^T \prod\limits_{-m \le j \le m, j \ne 0} P(w_{t+j}|w_t)
$$

### 1.2 对数似然函数

$$
    \sum\limits_{t=1}^T \sum\limits_{-m \le j \le m, j \ne 0} \log P(w_{t+j}|w_t)
$$

### 1.3 求梯度
- 于是对应的损失函数为:  

$$
    - \sum\limits_{t=1}^T \sum\limits_{-m \le j \le m, j \ne 0} \log P(w_{t+j}|w_t)
$$

- 接下来要对整个损失函数求导
    1. 先对 $\log P(w_o|w_c)$ 求导
        > $P(w_o|w_c)$ 即中心词为$w_c$，上下文中出现$w_o$的条件概率

        $$
        \begin{aligned}
            \frac {\partial \log P(w_o|w_c)} {\partial v_c}
            & = \frac {\partial} {\partial v_c} \left( \log \frac {exp( u_o^T v_c)} {\sum\limits_i exp(u_i^T v_c)} \right) \\
            & = \frac {\partial} {\partial v_c} \left( u_o^T v_c - \log \sum\limits_i exp(u_i^T v_c) \right) \\
            & = u_o - \frac 1 {\sum\limits_i exp(u_i^T v_c)} \frac {\partial} {\partial v_c} \left( \sum\limits_i exp(u_i^T v_c) \right) \\
            & = u_o - \frac 1 {\sum\limits_i exp(u_i^T v_c)} \frac {\partial \left( exp(u_0^T v_c) + exp(u_1^T v_c) + \cdots + exp(u_n^T v_c) \right)} {\partial v_c}  \\
            & = u_o - \frac 1 {\sum\limits_i exp(u_i^T v_c)} \left( exp(u_0^T v_c) u_0 + exp(u_1^T v_c) u_1 + \cdots + exp(u_n^T v_c) u_n \right) \\
            & = u_o - \left( \frac {exp(u_0^T v_c)} {\sum\limits_i exp(u_i^T v_c)} u_0 + \frac {exp(u_1^T v_c)} {\sum\limits_i exp(u_i^T v_c)} u_1 + \cdots + \frac {exp(u_n^T v_c)} {\sum\limits_i exp(u_i^T v_c)} u_n \right) \\
            & = u_o - \sum\limits_k \frac {exp(u_k^T v_c)} {\sum\limits_i exp(u_i^T v_c)} u_k \\
            & = u_o - \sum\limits_k P(w_k|w_c) u_k \\
            & = observed - expected
        \end{aligned}
        $$
    2. 计算整个式子对 $v_t$ 的偏导
        - 只需将长度为T的文本序列中，所有以$w_t$为中心词的条件概率加和，然后对每一项求导
    3. 对其他参数 $u_i、v_i$ 的求导同理

## 2. CBOW(连续词袋模型)
> 根据上下文预测中心词

### 2.1 概率解释
1. 类似的，指定上下文窗口为m，给定上下文，预测中心词的概率为:

    $$
        P(w_c|w_{c-m}, \cdots, w_{c-1}, w_{c+1}, \cdots, w_{c+m}) = \frac {exp( u_c^T \bar{v}_o)} {\sum\limits_i exp(u_i^T \bar{v}_o)}
    $$

    > 其中 $\bar{v}_o = \frac 1 {2m} \left( v_{c-m} + \cdots + v_{c-1} + v_{c+1} + \cdots + v_{c+m} \right)$

2. 给定一个长度为T的文本序列，设时间步t的词为 $w_t$ :

$$
    \prod\limits_{t=1}^T P(w_t|w_{t-m}, \cdots, w_{t-1}, w_{t+1}, \cdots, w_{t+m})
$$

### 2.2 对数似然函数

$$
    \sum\limits_{t=1}^T \log P(w_t|w_{t-m}, \cdots, w_{t-1}, w_{t+1}, \cdots, w_{t+m})
$$

### 2.3 求梯度
- 于是对应的损失函数为:  

$$
    - \sum\limits_{t=1}^T \log P(w_t|w_{t-m}, \cdots, w_{t-1}, w_{t+1}, \cdots, w_{t+m})
$$

- 求导略...

## 3. 优化softmax
> 在[这一节](神经网络/自然语言处理/word2vec?id=_13-求梯度)求梯度的过程中，我们知道，对一个长文本中的每个词，都需要计算 $\sum\limits_k P(w_k|w_c)$  
> 词表比较大时，这个计算量非常大，因此提出了两种优化策略

### 3.1 Hierarchical Softmax(分层softmax)
相当于把`n分类`问题 转化为 $\log_2 n$ 个`二分类`问题

### 3.2 Negative Sampling(负采样)
当中心词为$w_c$时，在其上下文中出现的词称为`正样本`，未出现过的词称为`负样本`  
求导时计算量大，是因为我们在计算条件概率 $P(w_o|w_c)$ 时使用了整个词表:
> 回忆下公式
$$P(w_o|w_c) = \frac {exp( u_o^T v_c )} {\sum\limits_i exp(u_i^T v_c)}$$
> 这里的 $\sum\limits_i$ 就包含了所有的`正样本`、`负样本`

**负采样**的思想是，**每次在计算 $P(w_o|w_c)$ 时，只使用一小部分`负样本`**

具体计算时，将$P(w_o|w_c)$拆解为:  
P(出现`正样本`$w_o$) * P(出现`负样本`$w_0$) * ... * P(出现`负样本`$w_k$)  
对应数学公式如下:  
$$
    P(w_o|w_c) = \sigma(u_o^T v_c) \prod\limits_{k \ne o} (1-\sigma(u_k^T v_c))
$$

## 4. 代码实现
> 完整代码请参照`《dive into DL pytorch》`[相应章节](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter10_natural-language-processing/10.3_word2vec-pytorch)
- 基于**负采样**的**skip-gram**模型

1. 首先定义模型、损失函数、优化算法
```
# 定义模型
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    ‘’‘
    center为中心词，(batch_size, 1)
    contexts_and_negatives为中心词所对应的正、负样本，(batch_size, 60)
    embed_v、embed_u为模型参数
    ’‘’

    v = embed_v(center) # (batch_size, 1, embedding_dim)
    u = embed_u(contexts_and_negatives) # (batch_size, 60, embedding_dim)
    pred = torch.bmm(v, u.permute(0, 2, 1)) # (batch_size, 1, 60)
    
    return pred

# 初始化网络参数
embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)

# 定义损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)

loss = SigmoidBinaryCrossEntropyLoss()

# 定义优化算法: Adam
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
```

2. 训练过程
```
def train(net, lr, num_epochs):
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for center, context_negative, mask, label in data_iter:
            pred = skip_gram(center, context_negative, net[0], net[1])  # 正向传播，预测

            # 使用掩码变量mask来避免填充项对损失函数计算的影响
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean() # 一个batch的平均loss

            optimizer.zero_grad() # 梯度清零
            l.backward() # 反向传播，计算梯度
            optimizer.step() # 更新参数
            
            # 统计损失
            l_sum += l.item()
            n += 1

        # 每轮打印信息
        print('epoch %d, loss %.2f, time %.2fs' % (epoch + 1, l_sum / n, time.time() - start))
```

## 5. GloVe
1. **共现矩阵**+**SVD**
    - **word2vec**出现之前，也有一些其他方法来获取词向量，其中一种就是**共现矩阵**+**SVD**(降维)
    - 相比**word2vec**，有效利用统计信息，训练速度很快，方法简单
    - 但无法捕捉词语相似度
2. GloVe
    - 于是提出了Global Vector，结合两者的优势
    - 只需要遍历一遍预料，统计**共现矩阵**

## 6. 效果评测

### 6.1 内部评测
1. 语义类比
    - 例如 man:women 相当于 king:?
2. 语法对比
    - 例如 strong:stronger:strongest
3. 词语相似度

### 6.2 外部评测
直接评测在下游任务上的表现，例如**NER**(命名实体识别)