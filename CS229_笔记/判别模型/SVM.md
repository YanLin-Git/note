# SVM(支持向量机)

## 1. 目标函数 (最优边界分类器)
在**感知机**中，我们用到了**函数间隔**，记为$\hat{\gamma}$，我们有:
$$
    \hat{\gamma} = y^{(i)}(w^T x^{(i)} + b)
$$

#### 1.1 函数间隔--->几何间隔：
对于同一个分离超平面，我们可以写成多种形式:
$$
w^T x + b = 0\\
2w^T x + 2b = 0\\
10w^T x + 10b = 0\\
\vdots
$$

那么同一个点(x, y)，对于该**分离超平面**，就会计算出 多个 $\hat{\gamma}$  
于是引出了**几何间隔**的概念，记作$\gamma$:
> 二维空间中，也就是我们高中学习的，点到直线的距离  
> 对于同一个点(x, y)，与一个**分离超平面**的**几何间隔**是固定的

$$
    \gamma = \frac {\hat{\gamma}} {||w||} = \frac {y^{(i)}(w^T x^{(i)} + b)} {||w||}
$$

#### 1.2 最大间隔分离超平面
在**感知机**中，我们只要求 对于一个线性可分的数据集，找到可将其分开的一个**分离超平面**  
**SVM**中，不仅想要正确划分，还想要**几何间隔**最大  
**用数学公式表示出来就是**:  

$$
\begin{aligned}
    & max_{\gamma,w,b} \quad \quad \gamma & (1)\\
    & s.t. \quad \frac {y^{(i)}(w^Tx^{(i)}+b)} {||w||} \geq \gamma,\quad i=1,...,m & (2)\\
\end{aligned}
$$

对于(1)式，我们可以等价的写为:

$$
    max_{\gamma,w,b} \quad \quad \frac {\hat{\gamma}} {||w||}
$$

对于(2)式，将$||w||$移到右边:
$$
    y^{(i)}(w^Tx^{(i)}+b) \geq ||w|| \gamma =  \hat{\gamma},\quad i=1,...,m
$$

**因此我们又可以表述为**:  

$$
\begin{aligned}
    & max_{\hat{\gamma},w,b} \quad \quad \frac {\hat{\gamma}} {||w||} & (1')\\
    & s.t. \quad y^{(i)}(w^Tx^{(i)}+b) \geq \hat{\gamma},\quad i=1,...,m & (2')\\
\end{aligned}
$$

> 而这里$\hat{\gamma}$的取值，对不等式没有任何影响，例如:
$$
\begin{aligned}
    & max_{\hat{\gamma}=1,w,b} \quad \quad \frac 1 {||w||} & (3)\\
    & s.t. \quad y^{(i)}(w^Tx^{(i)}+b) \geq 1,\quad i=1,...,m & (4)\\
\end{aligned}\\
\\
\begin{aligned}\\
    & max_{\hat{\gamma}=2,w',b'} \quad \quad \frac 2 {||w'||} & (3')\\
    & s.t. \quad y^{(i)}(w'^Tx^{(i)}+b') \geq 2,\quad i=1,...,m & (4')\\
\end{aligned}
$$
> (3)(4)式，是在参数空间(w,b)中寻找最优解  
> (3')(4')式，是在参数空间(w',b')，也即(2w,2b)中寻找最优解  
> - 两者是完全等价的

**不妨令$\hat{\gamma}=1$，就将问题表述为**:  
$$
\begin{aligned}
    & max_{w,b} \quad \quad \frac 1 {||w||} & (1'')\\
    & s.t. \quad y^{(i)}(w^Tx^{(i)}+b) \ge 1,\quad i=1,...,m & (2'')
\end{aligned}
$$

**最终形式**:  
更进一步，求解 $max_{w,b} \quad \frac 1 {||w||}$ 与 求解 $min_{w,b} \quad \frac 1 2 ||w||^2$是等价的  
所以也可以这样表述:  
$$
\begin{aligned}
    & min_{w,b} \quad \quad \frac 1 2 {||w||^2} & (1)\\
    & s.t. \quad 1 - y^{(i)}(w^Tx^{(i)}+b) \le 0,\quad i=1,...,m & (2)
\end{aligned}
$$

- **这就得到了SVM中的最终目标函数**

## 2. 对偶优化问题

#### 2.1 拉格朗日函数
首先引入拉格朗日函数:  
$$L(w,b,\alpha) = \frac 1 2 {||w||^2} + \sum\limits_{i=1}^m \alpha_i  (1 - y^{(i)}(w^Tx^{(i)}+b)) \quad (5)$$

我们在上一节中要求解的目标，相当于求解: $min_{w,b} \quad max_\alpha \quad L(w,b,\alpha)$

#### 2.2 对偶问题
**原始问题**: 求解$min_{w,b} \quad max_\alpha \quad L(w,b,\alpha)$  
可以转化为 求解其**对偶问题**: $max_\alpha \quad min_{w,b} \quad L(w,b,\alpha)$  
> 这里需要参考**拉格朗日对偶性**

**求解过程**:  
1. 先求解 $min_{w,b} \quad L(w,b,\alpha)$

$$
\begin{aligned}
    \nabla_w L(w,b,\alpha)=w-\sum^m_{i=1}\alpha_i y^{(i)}x^{(i)} =0\\
    \Longrightarrow \quad w=\sum^m_{i=1}\alpha_i y^{(i)}x^{(i)} \quad & (6)\\
    \frac{\partial}{\partial b}L(w,b,\alpha)=\sum^m_{i=1}\alpha_i y^{(i)}=0 \quad & (7)
\end{aligned}
$$
将(6)、(7)式代入(5)，简化一下，可以得到:  
$$L(w,b,\alpha)=L(\alpha)=\sum^m_{i=1}\alpha_i-\frac12 \sum^m_{i,j=1} y^{(i)}y^{(j)}\alpha_i\alpha_j(x^{(i)})^Tx^{(j)}$$

2. 再来求解 $max_\alpha \quad L(\alpha)$，就得到了相应的**对偶优化问题**

$$
\begin{aligned}
    max_\alpha \quad & W(\alpha) =\sum^m_{i=1}\alpha_i-\frac12\sum^m_{i,j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j \langle x^{(i)} x^{(j)}	\rangle & (8)\\
    s.t. \quad &  \alpha_i \geq 0,\quad i=1,...,m \quad & (引入拉格朗日函数时的约束)\\
    & \sum^m_{i=1} \alpha_iy^{(i)}=0 \quad & (上面的(7)式)
\end{aligned}
$$

**结论**:  
通过上面的过程，我们把原始问题: 求解$min_{w,b} \quad max_\alpha \quad L(w,b,\alpha)$  
转化为了求解 $max_\alpha \quad  W(\alpha) $
> 这是一个以$\alpha_i$为参数的最大值问题

## 3. 分离超平面的对偶形式
如果我们已经找出了上一节中对偶问题的最优解 $\alpha^\ast$，那么如何求解w、b，进而得到分离超平面?

设$\alpha^\ast = (\alpha_1^\ast,\alpha_2^\ast,\cdots,\alpha_n^\ast,)$是对偶问题的最优解  
可以按照下式求解$w^\ast，b^\ast$
1. $w^\ast=\sum\limits^m_{i=1} \alpha_i^\ast y^{(i)}x^{(i)}$
2. $(\alpha_1^\ast,\alpha_2^\ast,\cdots,\alpha_n^\ast,)$中一定存在下标j，使得$\alpha_j>0$, 可以这样求解$b^\ast$:  
$b^\ast = y_j - \sum\limits^m_{i=1} \alpha_i^\ast y^{(i)} \langle x^{(i)} x^{(j)} \rangle$

分离超平面$w^{\ast T} x + b^\ast = 0$就可以用类似的形式表示出来:  
$$
[\sum\limits^m_{i=1} \alpha_i^\ast y^{(i)}x^{(i)}] x + b^\ast = 0 \\
\Rightarrow \sum\limits^m_{i=1} \alpha_i^\ast y^{(i)} \langle x^{(i)} x \rangle + b^\ast = 0
$$

## 4. 核方法
可以看到，在对偶优化问题，以及得出分离超平面的过程中，我们只依赖于特征x之间的内积 $\langle x^{(i)} x^{(j)} \rangle$  
核方法将充分利用这一性质，使得算法能够在高纬空间中有效学习

举个例子:  
$$
\begin{aligned}
    K(x,z) &= (x^Tz)^2\\
    &= (\sum^n_{i=1}x_iz_i)(\sum^n_{j=1}x_jz_j)\\
    &= \sum^n_{i=1}\sum^n_{j=1}x_ix_jz_iz_j\\
    &= \sum^n_{i,j=1}(x_ix_j)(z_iz_j)\\
    &= (x_1 x_1, x_1 x_2, \cdots, x_n x_n)
        \begin{pmatrix}
            z_1 z_1\\
            z_1 z_2\\
            \vdots\\
            z_n z_n
        \end{pmatrix} \\
    &= \phi(x)^T \phi(z)
\end{aligned}
$$

在这个例子中，我们可以得到:

$$
\phi(x) = 
\begin{pmatrix}
    x_1 x_1\\
    x_1 x_2\\
    \vdots\\
    x_n x_n
\end{pmatrix} \\
$$

当我们在原始的n维空间中，使用特征$(x_1,x_2,\cdots,x_n)$学习时，使用的是$\langle x^{(i)} x^{(j)} \rangle$，即$x^T x$  
如果想在更高维，新的特征空间$(x_1 x_1, x_1 x_2, \cdots, x_n x_n)$中学习，则需要使用$\phi(x)^T \phi(x)$  
以n=3为例，原始特征空间3维，新的特征空间9维，直接计算的话，复杂度是$O(n^2)$  
而通过核函数，我们不需要去计算$\phi(x)^T \phi(x)$，直接使用K(x,x)即可  

**结论**:  
也就是说，我们把之前使用的$\langle x^{(i)} x^{(j)} \rangle$，都替换成K(x,x)，就相当于在一个高纬空间中去寻找分离超平面，而计算量没有明显变化

**常用核函数**:  
1. 多项式核：$K(x,z) = (x^Tz+c)^2$，c为常数
2. 高斯核：$K(x,z)=\exp (- \frac{\parallel x-z\parallel ^2}{2\sigma^2 })$
3. 字符串核函数

## 5. 线性不可分的情况
对于每一个样本点，我们添加了一个参数$\xi_i$，将原始问题改写为:

$$
\begin{aligned}
    min_{\gamma,w,b} \quad & \frac 12 \parallel w\parallel ^2+C\sum^m_{}\xi_i \\
    s.t. \quad & y^{(i)}(w^Tx^{(i)}+b) \geq1-\xi_i,i=1,...,m\\
    & \xi_i \geq 0 ,i=1,...,m
\end{aligned}
$$

这样就允许数据集里面有**函数间隔**小于1的情况了，一个样本的**函数间隔**可以为 $1 - \xi_i$ (其中 $\xi \geq  0$)  
这就需要我们给出 $C\xi_i$ 作为**目标函数成本函数的降低值**  
$C$ 是一个参数，用于控制相对权重，具体的控制需要在一对目标之间进行考量  
一个是使得 $\parallel w\parallel ^2$ 取最小值  
另一个是想让$\xi_i$尽可能小，确保绝大部分的样本都有至少为 1 的函数边界

同样推导一下对应的 对偶优化问题，可以得到:

$$
\begin{aligned}
    \max_\alpha \quad & W(\alpha) =\sum^m_{i=1}\alpha_i-\frac12\sum^m_{i,j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j \langle  x^{(i)},x^{(j)} \rangle \\
    s.t. \quad & 0\leq \alpha_i \leq C, \quad i=1,...,m\\
    & \sum^m_{i=1}\alpha_iy^{(i)}=0  \\
\end{aligned}
$$

对比第二节得出的结果，相当于添加了一组约束条件:  
$$ \alpha_i \leq C,\quad i=1,...,m $$  
C，则是我们自己调节的一个超参数

## 6. SMO优化算法(sequential minimal optimization)
#### 6.1 坐标上升法
要解决下面这样的无约束优化问题：

$$
\max_\alpha W(\alpha_1,\alpha_2,...,\alpha_m)
$$

我们采用下面的算法:

$$
\begin{aligned}
    &循环直至收敛：\{  \\
    &\qquad For\quad i=1,...,m, \{ \\
    &\qquad\qquad\alpha_i:= \arg \max_{\hat \alpha_i}W(\alpha_1,...,\alpha_{i-1},\hat\alpha_i,\alpha_{i+1},...,\alpha_m) \\
    &\qquad\} \\
    &\}
\end{aligned}
$$

坐标上升法的思想比较简单:  
如上式中所示，算法最内层的循环中，每次只调整一个(或者一批)参数 $\alpha_i$  
先假设除了某个特定的 $\alpha_i$ 之外的参数为常数，通过优化$W$来调整参数 $\alpha_i$

#### 6.2 SMO
在上面的版本中，最内层循环对变量重新优化的顺序是按照参数排列次序:  
$\alpha_1, \alpha_2, \cdots, \alpha_m, \alpha_1, \alpha_2, \cdots$ 来进行的
更进一步，我们可以根据预测哪个参数可以使 $W(\alpha)$ 增加最多，来选择下一个更新的参数

SMO算法，简单内容如下:  
重复直到收敛 {

1.  选择出某一对的 $\alpha_i$ 和 $\alpha_j$ (可以使 $W(\alpha)$ 增加最多的一对参数) 

2.  保持其他的 $\alpha_k$ 值固定($k\neq i,j$)，通过优化$W(\alpha)$，来调整 $\alpha_i$ 、 $\alpha_j$

}