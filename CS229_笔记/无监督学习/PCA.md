# PCA(主成分分析)
> 参考链接: https://zhuanlan.zhihu.com/p/77151308

设有 m 条 n 维数据
- 算法过程描述如下: 
    1. 将原始数据按列组成 n 行 m 列矩阵 X
    2. 去中心化: 将 X 的每一行进行零均值化，即减去这一行的均值
    3. 求出协方差矩阵 $C = \frac 1 m X X^T$ 
    4. 求出协方差矩阵的特征值及对应的特征向量
    5. 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 k 行组成矩阵 P
    6. Y=PX 即为降维到 k 维后的数据

## 额外补充
- 上述算法过程中的3、4步，也可以使用SVD来计算，实际上在sklearn中，就是用SVD求解的，原因有:
    - 当样本维度很高时，协方差矩阵计算太慢
    - 方阵特征值分解计算效率不高；
    - SVD 除了特征值分解这种求解方式外，还有更高效更准球的迭代求解方式，避免了$A^TA$的计算
    - 其实 PCA 与 SVD 的右奇异向量的压缩效果相同

## python实现，简单版本
```python
    import numpy as np

    def PCA(X,k):
        n_samples, n_features = X.shape

        # 去中心化
        mean = np.array([np.mean(X[:,i]) for i in range(n_features)])
        norm_X = X - mean

        # 计算协方差矩阵
        scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
        
        # 计算特征值、特征向量
        eig_val, eig_vec = np.linalg.eig(scatter_matrix)

        # 排序后，取出topk个特征向量
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
        eig_pairs.sort(reverse=True)
        feature = np.array([ele[1] for ele in eig_pairs[:k]])
        
        # 计算降维后到数据
        data = np.dot(norm_X, np.transpose(feature))

        return data
```