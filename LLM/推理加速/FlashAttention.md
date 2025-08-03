# FlashAttention
> 参考文章: https://www.zhihu.com/question/611236756/answer/3132304304

## 一、softmax

1. 原始的softmax公式:
$$
softmax(x_i) = \frac {e^{x_i}} {\sum\limits_j e^{x_j}}
$$
2. 稳定版本:
$$
m(x) = max(x_1, x_2, \dots, x_n) \\
softmax(x_i) = \frac {e^{x_i - m(x)}} {\sum\limits_j e^{x_j - m(x)}}
$$
3. 详细计算流程:
$$
\begin{aligned}
m(x) &= max(x_1, x_2, \dots, x_n) \quad & (1) \\
f(x) &= (e^{x_1 - m(x)}, e^{x_2 - m(x)}, \dots, e^{x_n - m(x)}) \quad & (2) \\
l(x) &= \sum\limits_i f(x)_i \quad & (3) \\
softmax(x) &= \frac {f(x)} {l(x)} \quad & (4)
\end{aligned}
$$

## 二、softmax的动态更新
> 直接使用上述计算公式的话，首先需要将整个向量都加载到内存  
> 假设我们的向量有1000000维，但是计算资源有限，一次只能加载一部分，又该怎么计算呢？

1. 首先将向量分块，写成这种形式:
    $$
    (\underbrace{x_1^{(1)}, x_2^{(1)}, \dots, x_B^{(1)}}_{第1块}, \underbrace{x_1^{(2)}, x_2^{(2)}, \dots, x_B^{(2)}}_{第2块}, \dots, \underbrace{x_1^{(n)}, x_2^{(n)}, \dots, x_B^{(n)}}_{第n块})
    $$
2. 将第一块$(x_1^{(1)}, x_2^{(1)}, \dots, x_B^{(1)})$加载进来，按照该流程来计算:
    $$
    \begin{aligned}
    m(x^{(1)}) &= max(x_1^{(1)}, x_2^{(1)}, \dots, x_B^{(1)}) \quad & (1) \\
    f(x^{(1)}) &= (e^{x_1^{(1)} - m(x^{(1)})}, e^{x_2^{(1)} - m(x^{(1)})}, \dots, e^{x_B^{(1)} - m(x^{(1)})}) \quad & (2) \\
    l(x^{(1)}) &= \sum\limits_i f(x^{(1)})_i \quad & (3) \\
    softmax(x^{(1)}) &= \frac {f(x^{(1)})} {l(x^{(1)})} \quad & (4)
    \end{aligned}
    $$
    - 并且保存(1)、(3)两式计算出来的标量(**这两个标量会常驻内存**)
        $$
        m_{max} = m(x^{(1)}) \\
        l_{all} = l(x^{(1)})
        $$

    > $m(x^{(1)})$、$f(x^{(1)})$、$l(x^{(1)})$，不再使用，就释放掉  
    > $softmax(x^{(1)})$ 可以理解为 暂时把它写入硬盘

3. 接下来将第二块$(x_1^{(2)}, x_2^{(2)}, \dots, x_B^{(2)})$加载进来，也按照该流程来计算:
    $$
    \begin{aligned}
    m(x^{(2)}) &= max(x_1^{(2)}, x_2^{(2)}, \dots, x_B^{(2)}) \quad & (1) \\
    f(x^{(2)}) &= (e^{x_1^{(2)} - m(x^{(2)})}, e^{x_2^{(2)} - m(x^{(2)})}, \dots, e^{x_B^{(2)} - m(x^{(2)})}) \quad & (2) \\
    l(x^{(2)}) &= \sum\limits_i f(x^{(2)})_i \quad & (3) \\
    softmax(x^{(2)}) &= \frac {f(x^{(2)})} {l(x^{(2)})} \quad & (4)
    \end{aligned}
    $$
    - 然后来更新这两个标量
        $$
        m_{max}^{new} = max[m_{max}, m(x^{(2)})] \\
        l_{all}^{new} = e^{m_{max}-m_{max}^{new}} l_{all} + e^{m(x^{(2)})-m_{max}^{new}} l(x^{(2)})
        $$

    > 此时$m(x^{(2)})$、$f(x^{(2)})$、$l(x^{(2)})$、$softmax(x^{(2)})$还没释放

4. 更新$softmax(x^{(2)})$
    $$
    \begin{aligned}
    f^{new}(x^{(2)}) &= f(x^{(2)}) e^{m(x^{(2)})-m_{max}^{new}} \\
    softmax^{new}(x^{(2)}) &= \frac {f^{new}(x^{(2)})} {l_{all}^{new}}
    \end{aligned}
    $$
    > $m(x^{(2)})$、$f(x^{(2)})$、$l(x^{(2)})$，不再使用，就释放掉  
    > $softmax(x^{(2)})$更新完成，可以暂时把它写入硬盘

5. 更新$softmax(x^{(1)})$

    > 先把之前计算好的$softmax(x^{(1)})$加载进来

    $$
    \begin{aligned}
    f^{new}(x^{(1)}) &= \underbrace{softmax(x^{(1)}) l_{all}}_{即f(x^{(1)})} e^{m_{max}-m_{max}^{new}} \\
    softmax^{new}(x^{(1)}) &= \frac {f^{new}(x^{(1)})} {l_{all}^{new}}
    \end{aligned}
    $$
    > $softmax(x^{(1)})$更新完成，可以暂时把它写入硬盘

6. 接下来将第三块$(x_1^{(3)}, x_2^{(3)}, \dots, x_B^{(3)})$加载进来，继续往下计算
    1. 计算$softmax(x^{(3)})$，然后更新
    2. 将之前计算好的$softmax(x^{(1)})$、$softmax(x^{(2)})$加载进来，更新

## 三、FlashAttention2中的改进
> 为减少计算量，FlashAttention2做了进一步改进

1. 将向量分块
2. 将第一块$(x_1^{(1)}, x_2^{(1)}, \dots, x_B^{(1)})$加载进来，按照该流程来计算:
    $$
    \begin{aligned}
    m(x^{(1)}) &= max(x_1^{(1)}, x_2^{(1)}, \dots, x_B^{(1)}) \quad & (1) \\
    f(x^{(1)}) &= (e^{x_1^{(1)} - m(x^{(1)})}, e^{x_2^{(1)} - m(x^{(1)})}, \dots, e^{x_B^{(1)} - m(x^{(1)})}) \quad & (2) \\
    l(x^{(1)}) &= \sum\limits_i f(x^{(1)})_i \quad & (3)
    \end{aligned}
    $$
    - 并且保存(1)、(3)两式计算出来的标量(**这两个标量会常驻内存**)
        $$
        m_{max} = m(x^{(1)}) \\
        l_{all} = l(x^{(1)})
        $$

    > $m(x^{(1)})$、$l(x^{(1)})$，不再使用，就释放掉  
    > $f(x^{(1)})$ 可以理解为 暂时把它写入硬盘  
    > 可以看到这里没计算 $softmax(x^{(1)})$，只是把 $f(x^{(1)})$ 这个中间计算结果，暂时保存起来

3. 接下来将第二块$(x_1^{(2)}, x_2^{(2)}, \dots, x_B^{(2)})$加载进来
    $$
    \begin{aligned}
    m(x^{(2)}) &= max(x_1^{(2)}, x_2^{(2)}, \dots, x_B^{(2)}) \quad & (1) \\
    m_{max}^{new} &= max[m_{max}, m(x^{(2)})] \quad 更新m_{max} \\
    f(x^{(2)}) &= (e^{x_1^{(2)} - m_{max}^{new}}, e^{x_2^{(2)} - m_{max}^{new}}, \dots, e^{x_B^{(2)} - m_{max}^{new}}) \quad & (2) \\
    l(x^{(2)}) &= \sum\limits_i f(x^{(2)})_i \quad & (3) \\
    \end{aligned}
    $$
    - 然后来更新$l_{all}$
        $$
        l_{all}^{new} = e^{m_{max}-m_{max}^{new}} l_{all} + l(x^{(2)})
        $$

    > 这里计算顺序做了调整，先更新$m_{max}$，再去计算$f(x^{(2)})$、$l(x^{(2)})$  
    > 这样就不用再去更新$f(x^{(2)})$、$l(x^{(2)})$
4. 更新$f(x^{(1)})$

    > 先把之前计算好的$f(x^{(1)})$加载进来

    $$
    f^{new}(x^{(1)}) = f(x^{(1)}) e^{m_{max}-m_{max}^{new}}
    $$

    > $f(x^{(1)})$更新完成，可以暂时把它写入硬盘
5. 接下来将第三块$(x_1^{(3)}, x_2^{(3)}, \dots, x_B^{(3)})$加载进来，继续往下计算
    1. 计算$f(x^{(3)})$
    2. 将之前计算好的$f(x^{(1)})$、$f(x^{(2)})$加载进来，更新
6. 最后再计算softmax(x)
    > 我们已经计算好了所有的 $f(x^{(1)})$、$f(x^{(2)})$、...、$f(x^{(n)})$  
    > 也获取到了全局的 $l_{all}$

    - 于是直接计算就好:
    $$
    softmax(x) = \frac {f(x)} {l_{all}}
    $$

## 四、完整计算流程
> 梳理完softmax的分块计算，基本就理解FlashAttention了  
> flashattn的输入是QKV矩阵，输出O矩阵  
> 接下来要做的就是理解 如何将QKVO矩阵分块，每次去计算一小部分O矩阵，再去动态更新
