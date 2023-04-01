- 按照这个顺序梳理:
```mermaid
graph LR
    GDA[1 高斯判别模型]
    Bayes[2 朴素贝叶斯]
    HMM[3 HMM]

    SCMX[生成模型] --- GDA & Bayes & HMM
    GDA -.属于.- LJ[逻辑回归]
    
```