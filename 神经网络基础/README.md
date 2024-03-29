# 神经网络基础
> 这部分内容参考`CS224n`和 [Dive-into-DL-PyTorch](https://tangshusen.me/Dive-into-DL-PyTorch/#/)

1. 基础组件
    - 先熟悉这些基础组件，有了积木，就可以开始搭建各种模型
    1. [RNN](神经网络基础/RNN.md)
    2. [GRU](神经网络基础/GRU.md)
    3. [LSTM](神经网络基础/LSTM.md)
    4. [CNN](神经网络基础/CNN.md)
    5. [transformer](神经网络基础/transformer.md)
2. 常见概念
    1. [损失函数](神经网络基础/损失函数.md) ----- 定义优化目标
    2. [优化算法](神经网络基础/优化算法.md) ----- 有了优化目标后，如何求解
    3. [学习率调整](神经网络基础/学习率调整.md) ----- 训练时，让学习率随着步数调整
    4. [激活函数](神经网络基础/激活函数.md) ----- 引入非线性
    5. [正则](神经网络基础/正则.md) ----- 防止过拟合，提升单个模型的性能
    6. [前、后向传播](神经网络基础/前、后向传播.md)
        - 实现一个softmax回归，关注其中的`forward()`、`backward()`
3. 训练流程
    - 了解前面这些概念后，就可以实现一个完整的训练流程
    1. [trainer](神经网络基础/trainer.md)