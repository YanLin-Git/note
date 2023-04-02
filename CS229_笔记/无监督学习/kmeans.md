# kmeans

- 算法内容如下:

&emsp; 1. 随机初始化k个**聚类重心**: $\mu_1, \mu_2, \cdots, \mu_k \in R^n$

&emsp; 2. 重复直到收敛 {

&emsp;&emsp; 将每个训练样本$x^{(i)}$ “分配”给距离最近的**聚类重心**$\mu_j$

&emsp;&emsp; 把每个聚类重心$\mu_j$ 移动到所分配的样本点的均值位置

&emsp; }

