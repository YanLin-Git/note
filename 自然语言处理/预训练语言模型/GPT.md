# GPT系列

## GPT-1
### 1. 模型结构
- **transformer**的`decoder`

### 2. 预训练任务
标准的语言模型，给定一个文本序列，根据前边 $i-1$ 个词，来预测第 $i$ 个词

### 3. 下游任务上的fine-tuning
- 在这之前的**word2vec**、**ELMo**，采用的feature-based的预训练模式
    > 先在一个大规模预料上去预训练，得到word_embdding这种特征  
    > 然后针对具体的下游任务，直接使用训练好的特征，构建各自的模型去训练
- **GPT**首次提出这种fine-tuning模式
    > 针对具体的下游任务，只需要修改`input的形式`，就可以套用预训练所使用的模型
- 具体修改方式可以参照论文中这张图片:
    ![GPT_1_fine_tuning.jpg](jpgs/GPT_1_fine_tuning.jpg) 

### 4. zero-shot
> - paper: https://arxiv.org/abs/1710.04837  
> - 知乎介绍文章: https://zhuanlan.zhihu.com/p/34656727

论文中还做了实验，只使用预训练模型，不经过**fine-tuning**，直接去下游任务上评估效果  
发现随着预训练过程中模型的不断更新，下游任务上的表现也越好

## GPT-2
> 只是扩大了规模，效果上并不出彩  
> 相比GPT-1，更加强调**zero-shot**
- 在GPT的基础上，做了几处修改:
    1. 进一步增加预训练数据数量
        - **Bert**增加了`Wikipedia`，**GPT-2**整理出了`WebText`
    2. 加深transformer层次，参数规模比**GPT-1**大了一个数量级
    3. subword算法
        - BPE ---> BBPE
    4. 只使用**zero-shot**
        - 不再进行下游任务的**fine-tuning**，预训练的模型直接去下游任务上评估效果

## GPT-3
- 进一步扩大规模，效果也提上来了，引起广泛关注，例如后来的`提示学习`、`思维链`
- 对之前的**zero-shot**进一步探索，提出一个新名词 `in-context learning`，包括3种方式:
    1. zero-shot prompting
    2. one-shot prompting
    3. few-shot prompting

    > 预训练好语言模型之后，在做推理时，只需要将`input`修改为特定格式，经过模型，得到`output`

- 具体做法可以参照论文中这张图片:
    ![GPT_3_in_context_learning.jpg](jpgs/GPT_3_in_context_learning.jpg) 
