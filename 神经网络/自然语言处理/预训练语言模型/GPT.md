# GPT

# GPT-1
## 1. 模型结构
- **transformer**的`decoder`

## 2. 预训练任务
标准的语言模型，给定一个文本序列，根据前边 $i-1$ 个词，来预测第 $i$ 个词

## 3. 下游任务上的fine-tuning
- 在这之前的**word2vec**、**ELMo**，采用的feature-based的预训练模式
    > 先在一个大规模预料上去预训练，得到word_embdding这种特征  
    > 然后针对具体的下游任务，直接使用训练好的特征，构建各自的模型去训练
- **GPT**首次提出这种fine-tuning模式
    > 针对具体的下游任务，只需要修改`input的形式`，就可以套用预训练所使用的模型
- 具体修改方式可以参照论文中这张图片:

    ![GPT1_fine_tuning.jpg](jpgs/GPT1_fine_tuning.jpg) 

## 4. zero-shot
论文中还做了实验，只使用预训练模型，不经过**fine-tuning**，直接去下游任务上评估效果  
发现随着预训练过程中模型的不断更新，下游任务上的表现也越好