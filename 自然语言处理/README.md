# 自然语言处理

- 目录
    1. [常见数据集](自然语言处理/常见数据集.md)
    2. [subword](自然语言处理/subword.md)
    3. [预训练语言模型](自然语言处理/预训练语言模型/)
        - 梳理预训练语言模型的发展史，沿着这个脉络，阅读相关论文
    4. [LLM(大语言模型)](自然语言处理/LLM/)
        - 冲向大语言模型！！

- 几大模型规模对比:

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