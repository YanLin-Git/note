# LLM(大语言模型)

- 目录
    1. [instructGPT](自然语言处理/LLM/instructGPT.md)
    2. [peft](自然语言处理/LLM/peft.md)
    3. [langchain](自然语言处理/LLM/langchain.md)
    4. [Chinese-LLaMA-Alpaca](自然语言处理/LLM/Chinese-LLaMA-Alpaca.md)
    5. [ChatGLM](自然语言处理/LLM/ChatGLM.md)

1. **todo.list**
    1. Positional Embedding
        - 绝对位置编码
            1. 三角式
            2. 可学习式
        - 相对位置编码 (建模单词两两之间的相对距离)
            1. transformer-XL
            2. ALiBi (BLOOM、MPT采用)
            3. RoPE (旋转式编码，绝对编码实现相对编码， PaLM、GLM-130B、LLaMa采用)
    2. 混合精度训练 & 模型量化(quantization)
        1. Apex的amp
        2. bitsandbytes
        3. 例如ChatGLM2的P-Tuning，采用这种方式: `模型主体`部分，采用fp16；`P-Tuning`部分，采用fp32
        4. QLoRA
    3. 内存占用
        - 激活函数占大头
            - 以`bert-base`为例，model占用2%，optimizer占用10%，激活函数占用87.6%
            - 解决方式: 时间换空间，激活函数重演(rematerialization) 或者常见的 梯度检查点(gradient checkpoint)
    4. 多卡训练
        - 数据并行 (Data Parallelism)
        - 模型并行
            1. 简单的模型并行 (分层)
            2. 流水线并行 (Gpipe)
            3. 张量并行 (Megatron-LM)
        - 优化器并行 (ZeRO)
    5. 训练稳定性

2. 模型汇总
|模型|发布方|参数量|备注|
|---|---|---|---|
|LLaMA|Meta|65B|[paper](https://arxiv.org/pdf/2302.13971.pdf)|
|LLaMA2|Meta||[paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)|
|Alpaca|斯坦福|7B|效果可媲美175B的LLM|
|vicuna||||
|Chinese-LLaMA-Alpaca|哈工大||https://github.com/ymcui/Chinese-LLaMA-Alpaca|
|GLM|清华||https://github.com/THUDM/GLM|
|ChatGLM-6B|清华||https://github.com/THUDM/ChatGLM-6B|
|ChatGLM2-6B|清华||https://github.com/THUDM/ChatGLM2-6B|
|instructGPT|openAI||`paper`, chatGPT介绍中说，使用instructGPT论文里的方法|
|toolformer|||`paper`, 使得LLM能够使用各种工具|
|PaLM|google|||
|Chinchilla|DeepMind|||
|visual chatGPT||||
|Bard|google|||
|Claude|Anthropic||据说chatGPT的最大竞争对手|
|GPT-4|openAI|||
|Copilot|microsoft|||