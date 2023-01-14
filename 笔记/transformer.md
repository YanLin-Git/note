# transformer快速入门

## 一、参考文章
1. 入门: https://jalammar.github.io/illustrated-transformer/
    - 中文翻译版: https://zhuanlan.zhihu.com/p/54356280
2. 也可参考: https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3
    - 没去研究
3. 进阶版: http://nlp.seas.harvard.edu/2018/04/03/attention.html 
    - 熟悉完pytorch后再来看这里
4. github上另外一种实现: https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb

## 二、一些疑问
1. 前馈神经网络，就是一个全链接层
    - 猜测是为了引入了非线性
2. transformer中每层后面都会有一个归一化(normalize)
    > - 导读: https://zhuanlan.zhihu.com/p/38176412
    > - 各种normalize方法: https://zhuanlan.zhihu.com/p/43200897

    |normalization|备注|适用场景|
    |---|---|---|
    |batch normalization|对于同一个神经元，mini-batch中不同训练实例导致的不同激活|优先考虑|
    |layer normalization|同隐层的所有神经元|对于RNN，目前只有Layer有效|
    |instance normalization|CNN中卷积层的单个通道|GAN等图片生成类任务|
    |group normalization|layer和instance的折衷||

