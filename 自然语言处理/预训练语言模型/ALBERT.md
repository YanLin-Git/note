# ALBERT (a lite Bert)

在原始Bert的基础上，做了3处修改:
- 减少参数量
    1. word embedding层的矩阵分解
        > (vocab_size, hidden_size) ---分解为--> (vocab_size, emb_size) x (emb_size, hidden_size)
    2. 参数共享
- 提升性能
    3. 预训练的句子任务: SOP(sentence order prediction)
        > NSP中，两个连续句子为正例，随机选取的为负例，任务过于简单，模型学不到什么知识  
        > SOP中，两个连续句子为正例，交换两个句子顺序为负例
