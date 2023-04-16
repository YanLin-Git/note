# 预训练语言模型
> 参考: https://zhuanlan.zhihu.com/p/254821426  
> 梳理一下预训练模型的脉络，沿着这个脉络，一步步熟悉每个模型

1. 常见模型结构
    - transformer不同的用法
        1. Encoder
        2. Decoder
        3. Encoder + Decoder
    - 自监督的学习方法
        1. AutoEncoding(AE)
            - 即常见的双向语言模型
        2. AutoRegressive(AR)
            - 从左到右的单向语言模型

2. 常见模型
    1. Decoder + AR (适合做语言生成类任务)
        - GPT系列
            - GPT1、GPT2、GPT3
    2. Encoder + AE (适合做语言理解类任务)
        1. Bert
        2. RoBERTa (得到充分训练的Bert)
        3. ALBERT
        4. ELECTRA
    3. Encoder + Decoder (可以同时做生成类、理解类任务)
        1. MASS
        2. BART
        3. T5
    4. Prefix Language Model
        - UniLM
    5. Permuted Language Model
        - XLNet

3. 训练任务
    > 在研究每个模型细节时，顺便关注下训练任务的变化，先搬个结论:

    1. Bert中训练任务:
        1. 单词级任务
            - MLM，Mask语言模型
        2. 句子级任务
            - NSP，下一句预测
    2. 目前认为最有效的训练任务:
        1. 单词级任务
            - Span类任务，Mask的不是一个独立单词，而是连续的单词片断、短语等等
        2. 句子级任务
            - SOP(sentence order prediction)
                > NSP中，两个连续句子为正例，随机选取的为负例，任务过于简单，模型学不到什么知识  
                > SOP中，两个连续句子为正例，交换两个句子顺序为负例

4. 汇总
    > 综上，接下来这部分将按照以下顺序 熟悉模型 (前3个上边没提):  

    1. NNLM(2003) [paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
    2. word2vec(2013) [paper](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
    3. ELMo
    4. GPT
    5. Bert
    6. RoBERTa
    7. ALBERT
    8. ELECTRA
    9. MASS
    10. BART
    11. T5
    12. UniLM
    13. XLNet

