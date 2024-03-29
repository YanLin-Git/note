# subword算法

> 介绍NLP中常见的3种subword算法  
> 参考链接1: https://zhuanlan.zhihu.com/p/86965595  
> 参考链接2: https://zhuanlan.zhihu.com/p/198964217  
> 参考链接: https://blog.floydhub.com/tokenization-nlp/

## 1. BPE(byte pair encoding)
### 1.1 BPE
- **GPT-1**中使用
1. 首先对语料进行切词，统计词频，假设我们得到的结果为:
|word|频次|
|---|---|
|low|5|
|lower|2|
|newest|6|
|widest|3|

2. 在每个单词后加<\w>，如下:
|word|频次|
|---|---|
|low<\w>|5|
|lower<\w>|2|
|newest<\w>|6|
|widest<\w>|3|

3. 然后得到我们的初始化subword词表
|id|subword|频次|
|---|---|---|
|1|l|7|
|2|o|7|
|3|w|16|
|4|e|17|
|5|r|2|
|6|n|6|
|7|s|9|
|8|t|9|
|9|i|3|
|10|d|3|
|11|<\w>|16|

4. 合并其中最高频的 连续子词，这里就是`es`，频次9，更新subword词表如下
|id|subword|频次|
|---|---|---|
|1|l|7|
|2|o|7|
|3|w|16|
|4|e|17-9=8|
|5|r|2|
|6|n|6|
|7|s|9-9=0 `subword表中可以删掉这行`|
|8|t|9|
|9|i|3|
|10|d|3|
|11|<\w>|16|
|12|es|9|

5. 重复第4步，接下来就是`est`。然后一直重复，直到subword词表大小达到预期

- 每次合并后`subword词表`可能出现3种变化：
    1. +1，表明加入合并后的新字词，同时原来的2个子词还保留（2个字词不是完全同时连续出现）
    2. +0，表明加入合并后的新字词，同时原来的2个子词中一个保留，一个被消解（一个字词完全随着另一个字词的出现而紧跟着出现）
    3. -1，表明加入合并后的新字词，同时原来的2个子词都被消解（2个字词同时连续出现）
- 随着合并的次数增加，词表大小通常先增加后减小

### 1.2 BBPE (byte level BPE)
- **GPT-2**、**RoBERTa**中使用

对于英文单词，这样处理就够了，但其他语言，例如我们汉字，字数较多，在单字上去运行BPE，计算量比较大  
GPT的做法是在更小的单元byte上去运行BPE  
> 这样无论什么语言，统一用unicode编码，每个byte上，最多有$2^8=256$种可能

## 2. WordPiece
- **Bert**中使用，BPE的变种

在上面BPE的第四步合并时，不是选用`最高频`的连续子词，而是选择能够使`语言模型概率`提升最大的连续子词
> 例如一句话S=[t1,t2,t3,...,tn]，n个子词组成  
> 原始语言模型概率为 $P(s) = P(t_1)P(t_2)P(t_3)...P(t_n)$  
> 合并t1、t2后，语言模型概率为 $P(s) = P(t_1t_2)P(t_3)...P(t_n)$  

## 3. ULM(Unigram language Model)
- **ALBERT**、**XLNet**中使用
- 也是利用语言模型去建立`subword词表`，这里略

## 4. 如何使用这几种subword算法？
- 谷歌推出的开源工具[sentencepiece](https://github.com/google/sentencepiece)
1. 训练一个sentencepiece model
```python
import sentencepiece
# 使用BPE
sentencepiece.SentencePieceTrainer.train('--model_type=bpe --input=train.txt --model_prefix=bpe --vocab_size=500')
# 使用ULM
sentencepiece.SentencePieceTrainer.train('--model_type=unigram --input=train.txt --model_prefix=uni --vocab_size=500')
```
2. 加载
```python
import sentencepiece
sp_model = sentencepiece.SentencePieceProcessor()
sp_model.Load('./bpe.model')

# 顺便查看一下，有哪些词汇
vocabs = [sp_model.id_to_piece(id) for id in range(sp_model.get_piece_size())]
```
3. 序列化、反序列化
```python
sp_model.serialized_model_proto() # sp_model的序列化

from sentencepiece import sentencepiece_model_pb2
spm = sentencepiece_model_pb2.ModelProto()  
spm.ParseFromString( sp_model.serialized_model_proto() ) # 反序列化
spm.SerializeToString() # 序列化
```
