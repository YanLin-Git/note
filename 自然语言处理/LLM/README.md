# LLM(大语言模型)

- 目录
    1. [x] instructGPT [paper](https://arxiv.org/pdf/2203.02155v1.pdf)
    2. [x] peft

> GPT-3之后，出来一系列LLM，以及基于LLM的一些研究，遇到就放在这里  
1. 相关技术
    1. 模型架构
        - GPT: decoder
        - GLM: encoder-decoder
    2. Layer Norm
        - Post Layer Norm
        - Pre Layer Norm
        - Sandwich Layer Norm
    3. Positional Embedding
        - 绝对位置编码
            1. 三角式
            2. 可学习式
        - 相对位置编码 (建模单词两两之间的相对距离)
            1. transformer-XL
            2. ALiBi (BLOOM、MPT采用)
            3. RoPE (旋转式编码，绝对编码实现相对编码， PaLM、GLM-130B、LLaMa采用)
    4. 混合精度训练
        - forward时，采用fp16
        - backward时，采用fp32
    5. 内存占用
        - 激活函数占大头
            - 以`bert-base`为例，model占用2%，optimizer占用10%，激活函数占用87.6%
            - 解决方式: 时间换空间，激活函数重演(rematerialization)
    6. 模型并行
        - 数据并行 (Data Parallelism)
        - 张量并行
        - 流水线并行
    7. 训练稳定性
    8. 模型量化 (quantization)

2. 相关论文
|模型|发布方|参数量|备注|
|---|---|---|---|
|LLaMA|Facebook|65B||
|Alpaca|斯坦福|7B|效果可媲美175B的LLM|
|vicuna||||
|Chinese-LLaMA-Alpaca|||https://github.com/ymcui/Chinese-LLaMA-Alpaca|
|GLM|清华||https://github.com/THUDM/GLM|
|ChatGLM-6B|||https://github.com/THUDM/ChatGLM-6B|
|instructGPT|openAI||chatGPT介绍中说，使用instructGPT论文里的方法|
|toolformer|||使得LLM能够使用各种工具|
|PaLM|google|||
|Chinchilla|DeepMind|||
|visual chatGPT||||
|Bard|google|||
|Claude|Anthropic||据说chatGPT的最大竞争对手|
|GPT-4|openAI|||
|Copilot|microsoft|||