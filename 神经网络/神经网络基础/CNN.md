# CNN(卷积神经网络)

## 1. pytorch实现，简单版本:
1. **二维卷积层**
> 深度学习中，卷积运算，实际是互相关运算

- 二维互相关运算代码实现:
    ```python
    import torch 
    from torch import nn
    
    def corr2d(X, K):
        h, w = K.shape # 获取卷积核形状
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) # 计算输出形状
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i: i + h, j: j + w] * K).sum() # 对应元素相乘后求和
        return Y
    ```

2. **二维池化层**
    ```python
    import torch
    from torch import nn

    def pool2d(X, pool_size, mode='max'):
        X = X.float()
        p_h, p_w = pool_size
        Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1) # 计算输出形状
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if mode == 'max':
                    Y[i, j] = X[i: i + p_h, j: j + p_w].max() # 取对应区域的最大值
                elif mode == 'avg':
                    Y[i, j] = X[i: i + p_h, j: j + p_w].mean()       
        return Y
    ```

## 2. pytorch中的调用:
1. 卷积层
    ```python
    # 一维卷积，textCNN中使用
    nn.Conv1d(in_channels=10, out_channels=5, kernel_size=3)

    # 二维卷积
    nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))


    # kerner为方阵时，两种方式等价:
    nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(3, 3), padding=(0, 1), stride=(3, 4))
    nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=3, padding=(0, 1), stride=(3, 4))
    ```

2. 池化层
    ```python
    # 一维
    nn.MaxPool1d(pool_size = 5)

    # 二维
    nn.MaxPool2d(pool_size = 3, padding=1, stride=2)
    nn.MaxPool2d(pool_size = (3,3), padding=1, stride=2)
    ```

## 3. 应用实例
`textCNN`两种实现方式:
1. [一维卷积实现](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter10_natural-language-processing/10.8_sentiment-analysis-cnn)
2. [二维卷积实现](https://github.com/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN.py)