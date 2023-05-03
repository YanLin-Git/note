# 预训练语言模型

- 目录
    1. [x] NNLM(2003) [paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
    2. [x] word2vec(2013) [paper](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
    3. [x] ELMo(2018) [paper](https://arxiv.org/pdf/1802.05365v2.pdf)
    4. [ ] GPT系列
        - [x] GPT(2018.6) [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
        - [x] GPT-2(2019.2) [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
        - [x] GPT-3(2020.7) [paper](https://arxiv.org/pdf/2005.14165v4.pdf)
        - [ ] GPT-4(2023.3) [paper](https://arxiv.org/pdf/2303.08774v3.pdf)
    5. [x] Bert(2018.10) [paper](https://arxiv.org/pdf/1810.04805v2.pdf)
    6. [x] RoBERTa(2019.7) [paper](https://arxiv.org/pdf/1907.11692v1.pdf)
    7. [x] ALBERT(2020.2) [paper](https://arxiv.org/pdf/1909.11942v6.pdf)
    8. [ ] ELECTRA(2020.3) [paper](https://arxiv.org/pdf/2003.10555v1.pdf)
    9. [ ] MASS(2019.5) [paper](https://arxiv.org/pdf/1905.02450v5.pdf)
    10. [ ] BART(2019.10) [paper](https://arxiv.org/pdf/1910.13461v1.pdf)
    11. [x] T5(2020.7) [paper](https://arxiv.org/pdf/1910.10683v3.pdf)
    12. [ ] UniLM(2019.10) [paper](https://arxiv.org/pdf/1905.03197v3.pdf)
    13. [ ] XLNet(2020.1) [paper](https://arxiv.org/pdf/1906.08237v2.pdf)


> 参考: https://zhuanlan.zhihu.com/p/254821426  
> 梳理一下预训练模型的脉络，然后沿着这个脉络(上面的`4-13`)，一步步熟悉每个模型

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

3. 预训练任务的变化
    |模型|token级任务|句子级别任务|备注|
    |---|---|---|---|
    |原始Bert|MLM|NSP||
    |RoBERTa|MLM||NSP作用不大，舍弃|
    |ALBERT|MLM|SOP|两个连续句子为正例，交换两个句子顺序为负例|
    |T5|span mask||Mask的不是一个独立token，而是连续的几个token|

4. 几个模型规模对比:

$$ ELMo \to GPT \to Bert \to GPT-2 \to RoBERTa \to GPT-3 $$

|模型|数据集|词量|subword|模型|参数量|备注|
|---|---|---|---|---|---|---|
|$ELMo$|1B Word Benchmark|10亿|||90M||
|$GPT$|BooksCorpus|8亿|BPE|L=12,H=768,A=12|110M|`Word Benchmark`中句子顺序随机打乱，无法建模句子之间的依赖关系，弃用|
|$BERT_{base}$|16G BooksCorpus + Wikipedia|8亿 + 25亿|wordpiece|L=12,H=768,A=12|110M|为了跟`GPT-1`做对比|
|$BERT_{large}$|同上|同上|wordpiece|L=24,H=1024,A=16|340M||
|GPT-2|40G WebText <br> 1. 过滤低质量文本<br> 2. 除去Wikipedia中的数据||BBPE|L=12,H=768|117M|为了跟`GPT-1`做对比|
|GPT-2|同上||BBPE|L=24,H=1024|345M|为了跟`Bert`做对比|
|$GPT-2$|同上||BBPE|L=48,H=1600|1542M||
|$RoBERTa$|16G BooksCorpus + Wikipedia <br> 76G CC-News <br> 38G OpenWebText <br> 31G Stories||BBPE||||
|$GPT-3$|Common Crawl <br> WebText2 <br> Books1 <br> Books2 <br> Wikipedia|||L=96,H=12288,A=96|175B||