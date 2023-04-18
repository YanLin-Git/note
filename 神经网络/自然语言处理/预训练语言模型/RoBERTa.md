# RoBERTa

- 在原始Bert模型的基础上，RoBERTa通过实验，做了几点改进：
    1. 进一步增加预训练数据数量，能够改善模型效果
    2. 单词级任务
        - 静态mask ---> 动态mask
        - 具体做法很简单，就是对每一行输入，做10遍不同的mask去训练
    3. 句子级任务
        - 拿掉预训练任务中的Next Sentence Prediction子任务，它不必要存在
    4. 增大预训练的每个Batch的Batch Size，或者预训练步数
        |batch_size|steps|备注|
        |---|---|---|
        |256|1M|$BERT_{base}$中使用|
        |2K|125K|相同计算成本下，效果更好|
        |8K|31K|相同计算成本|
    5. subword算法
        - wordpiece ---> BBPE
        - 这里有个[简要介绍](神经网络/自然语言处理/subword.md)