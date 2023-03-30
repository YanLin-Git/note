# CRF(条件随机场)
> 这部分CS229中没有介绍，参考李航老师的《统计学习方法》

## 1. **最大熵模型**
在分类问题中，我们用这个来预测条件概率$P(y|x)$:
$$ P(y|x) = \frac {exp \left( \sum\limits_{i=1}^k w_i f_i(x,y) \right)} {Z(x)} $$

## 2. 线性链条件随机场
> 最大熵模型，在序列标注问题上的推广

在序列标注问题中，我们需要预测一个 向量 $\vec{y} = (y_1, y_2, \cdots, y_n)$  
**2.1 假设每一个标签，只与输入x有关**  
我们也可以类似来预测$P(y_i|x)$:  
$$ 
P(y_i|x) = \frac {exp \left( \sum\limits_{j=1}^k \mu_j s_j(x,y_i) \right)} {规范化因子} 
= \frac {exp \left( \sum\limits_k \mu_k s_k(x,y_i) \right)} {规范化因子} 
$$
> 这里稍微修改了下符号，假设共有k个特征函数$s_j$，特征权重为$\mu_j$


**2.2 考虑相邻标签之间的影响**  
在线性链条件随机场中，某个时刻的标签$y_i$，不仅与输入x有关，还与上一时刻的标签$y_{i-1}$有关  
这个可以通过另外添加一组特征函数来建模:
$$ 
P(y_i|x) = \frac {exp \left( \sum\limits_k \mu_k s_k(x,y_i) + \sum\limits_l \lambda_l t_l(x,y_{i-1},y_i) \right)} {规范化因子} 
$$

**2.3 需要预测一个序列**  
更进一步，我们需要建模$P(y|x)$，于是:  
$$
\begin{aligned}
    P(y|x) &= \prod\limits_i P(y_i|x) \\
    &= \prod\limits_i \frac {exp \left( \sum\limits_k \mu_k s_k(x,y_i,i) + \sum\limits_l \lambda_l t_l(x,y_{i-1},y_i,i) \right)} {规范化因子} \\
    &= \frac {exp \left( \sum\limits_i [\sum\limits_k \mu_k s_k(x,y_i,i) + \sum\limits_l \lambda_l t_l(x,y_{i-1},y_i,i)] \right)} {规范化因子} \\
    &= \frac {exp \left( \sum\limits_{i,k} \mu_k s_k(x,y_i,i) + \sum\limits_{i,l} \lambda_l t_l(x,y_{i-1},y_i,i) \right)} {Z(x)}
\end{aligned}
$$

$Z(x)$的表达式也很好给出:  
$$Z(x) = \sum\limits_y exp \left( \sum\limits_{i,k} \mu_k s_k(x,y_i,i) + \sum\limits_{i,l} \lambda_l t_l(x,y_{i-1},y_i,i) \right)$$
> 例如$\vec{y} = (y_1, y_2, y_3)$，每个$y_i$有2种类别，$y_i \in (0,1)$  
> 那么这里的$\sum\limits_y共有2^3种情况$:  
> $\vec{y} = (0,0,0)、(0,0,0)、\cdots、(1,1,1)$

**2.4 化简**
1. 特征函数，我们可以统一用$f_k$来表示:
$$
    f_k(x,y_{i-1},y_i,i) = 
    \begin{cases}
        s_k(x,y_i,i), \quad & k = 1,2,\cdots,K_1 \\
        t_l(x,y_{i-1},y_i,i), \quad & k = K_1+1,K_1+2,\cdots,K_1+l
    \end{cases}\\
$$
> 这里，$f_k(x,y_{i-1},y_i,i) \in (0,1)$

然后对$f_k(x,y_{i-1},y_i,i)$汇总:  
$$
    f_k(x,y) = \sum\limits_i f_k(x,y_{i-1},y_i,i)
$$

2. 特征权重，统一用$w_k$来表示:  
$$
    w_k = 
    \begin{cases}
        \mu_k, \quad & k = 1,2,\cdots,K_1 \\
        \lambda_l, \quad & k = K_1+1,K_1+2,\cdots,K_1+l
    \end{cases}
$$

于是我们就可以得到:
$$
\begin{aligned}
    P(y|x) &=  \frac {exp \left( \sum\limits_{i,k} \mu_k s_k(x,y_i,i) + \sum\limits_{i,l} \lambda_l t_l(x,y_{i-1},y_i,i) \right)} {Z(x)} \\
    &= \frac {exp \left( \sum\limits_{i,k} w_k f_k(x,y_{i-1},y_i,i) \right)} {Z(x)} \\
    &= \frac {exp \left( \sum\limits_k w_k f_k(x,y) \right)} {Z(x)} \\
    Z(x) &= \sum\limits_y exp \left( \sum\limits_{i,k} \mu_k s_k(x,y_i,i) + \sum\limits_{i,l} \lambda_l t_l(x,y_{i-1},y_i,i) \right) \\
    &= \sum\limits_y exp \left( \sum\limits_k w_k f_k(x,y) \right)
\end{aligned}
$$

> 形式上与**最大熵模型**很像，但注意这里的$\vec{y}$是一个序列

## 3. 条件随机场的矩阵形式
> 开源工具CRF++，代码实现与这部分对应

**3.1 矩阵定义**  
现在有一个观测序列 $\vec{x} = (x_1,x_2,\cdots,x_i,\cdots,x_n)$  
对应的状态序列 $\vec{y} = (y_1,y_2,\cdots,y_i,\cdots,y_n)$  
在某一个时刻i，假设$y_{i-1} \in (1,2), y_i \in (1,2,3)$  
针对$x_i$，我们可以计算如下矩阵:  
$$
\begin{aligned}
    M_i(x) &= 
    \begin{pmatrix}
        m_{11} & m_{12} & m_{13}\\
        m_{21} & m_{22} & m_{23}\\
    \end{pmatrix} \\
    &=
    \begin{pmatrix}
        exp \left( \sum\limits_k w_k f_k(x,y_{i-1}=1,y_i=1,i) \right) & exp \left( \sum\limits_k w_k f_k(x,y_{i-1}=1,y_i=2,i) \right) & exp \left( \sum\limits_k w_k f_k(x,y_{i-1}=1,y_i=3,i) \right)\\
        exp \left( \sum\limits_k w_k f_k(x,y_{i-1}=2,y_i=1,i) \right) & exp \left( \sum\limits_k w_k f_k(x,y_{i-1}=2,y_i=2,i) \right) & exp \left( \sum\limits_k w_k f_k(x,y_{i-1}=2,y_i=3,i) \right)\\
    \end{pmatrix} \\
\end{aligned}
$$

**3.2 利用矩阵进行计算**  
现在有一个观测序列 $\vec{x} = (x_1,x_2,x_3)$  
对应的状态序列 $\vec{y} = (y_1,y_2,y_3), y_i \in (1,2)$  
另外添加两个状态$y_0=1, y_4=1$  
按照上面的方法，就可以计算出4个矩阵:  
$$
\begin{aligned}
    M_1(x) &= 
    \begin{pmatrix}
        a_{11} & a_{12}
    \end{pmatrix} \\
    M_2(x) &= 
    \begin{pmatrix}
        b_{11} & b_{12} \\
        b_{21} & b_{22} \\
    \end{pmatrix} \\
    M_3(x) &= 
    \begin{pmatrix}
        c_{11} & c_{12} \\
        c_{21} & c_{22} \\
    \end{pmatrix} \\
    M_4(x) &= 
    \begin{pmatrix}
        1\\
        1\\
    \end{pmatrix} \\
\end{aligned}
$$

- 然后就可以这样计算:
    - $\vec{y} = (1,1,1)$对应的**非规范化概率**为: $a_{11} b_{11} c_{11}$
    - $\vec{y} = (1,1,2)$对应的**非规范化概率**为: $a_{11} b_{11} c_{12}$
    - ...
    - $\vec{y} = (2,2,2)$对应的**非规范化概率**为: $a_{12} b_{22} c_{22}$
    - 规范化因子$Z(x) = M_1(x) M_2(x) M_3(x) M_4(x)$

## 4. 前向-后向算法

## 5. 优化算法

#### 4.1 改进的迭代尺度法
#### 4.2 拟牛顿法

## 6. 预测算法

**维特比算法**