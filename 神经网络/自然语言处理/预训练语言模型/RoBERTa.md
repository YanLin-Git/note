# RoBERTa

- 在原始Bert模型的基础上，RoBERTa通过实验，证明了如下几点：
    1. 进一步增加预训练数据数量，能够改善模型效果
    2. 延长预训练时间或增加预训练步数，能够改善模型效果
    3. 急剧放大预训练的每个Batch的Batch Size，能够明显改善模型效果
    4. 拿掉预训练任务中的Next Sentence Prediction子任务，它不必要存在
    5. 输入文本的动态Masking策略有帮助