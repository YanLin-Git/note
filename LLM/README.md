# LLM(大语言模型)

- 目录
    1. [混合精度训练](LLM/混合精度训练.md)
    2. [模型量化](LLM/模型量化.md)
    3. [peft](LLM/peft.md)
    4. [并行训练](LLM/并行训练.md)
    5. [instructGPT](LLM/instructGPT.md)
    6. [Chinese-LLaMA-Alpaca](LLM/Chinese-LLaMA-Alpaca.md)
    7. [ChatGLM](LLM/ChatGLM.md)
    8. [langchain](LLM/langchain.md)

1. 如何降低资源占用？
    > 随着模型规模扩大，需要寻找各种方法来减少资源需求
    
    <details>
    <summary><b>一、数据类型上的工作</b></summary>

    1. 新的数据类型
        |数据类型||备注|
        |---|---|---|
        |FP32|float32|全精度(4bytes)|
        |FP16|float16|半精度(2bytes)|
        |BF16|bfloat16|半精度(2bytes)|
        |...|||
    2. [混合精度训练](LLM/混合精度训练.md)
        - 部分操作使用FP16，部分操作使用FP32
    3. [模型量化(quantization)](LLM/模型量化.md)
        - 进一步压缩，使用1byte来存储参数

    </details>

    <details>
    <summary><b>二、减少可训练参数的数量</b></summary>

    1. [peft](LLM/peft.md)
        - 模型主体部分的参数不变，只添加少量adapter，通过微调这部分参数，来达到微调整个模型的效果
        - 目前常用的两种方式
            1. lora [paper](https://arxiv.org/abs/2106.09685)
            2. P-Tuning v2
    2. QLoRA
        - 两种方式的融合: `Quantization` + peft中的`lora`方式

    </details>

    <details>
    <summary><b>三、梯度方面</b></summary>

    1. gradient accumulate
        - 多个batch前向传播，计算loss后，累加在一起，再进行反向传播
    2. gradient checkpoint
        - 显存占用，激活函数占大头，以`bert-base`为例，model占用2%，optimizer占用10%，激活函数占用87.6%
        - 解决方式：时间换空间，又称 激活函数重演(rematerialization)

    </details>

    <details>
    <summary><b>四、多卡训练</b></summary>

    1. [x] 数据并行 (Data Parallelism)
    2. [ ] 模型并行
        1. 简单的模型并行 (分层)
        2. 流水线并行 [Gpipe](https://arxiv.org/abs/1811.06965v5)
        3. 张量并行 [Megatron-LM](https://arxiv.org/abs/1909.08053v4)
    3. [ ] 优化器并行 [ZeRO](https://arxiv.org/abs/1910.02054v3)

    </details>

2. **待归纳**
    1. Positional Embedding
        - 绝对位置编码
            1. 三角式
            2. 可学习式
        - 相对位置编码 (建模单词两两之间的相对距离)
            1. transformer-XL
            2. ALiBi (BLOOM、MPT采用)
            3. RoPE (旋转式编码，绝对编码实现相对编码， PaLM、GLM-130B、LLaMa采用)
    2. 训练稳定性

3. 模型汇总

    |模型|发布方|参数量|备注|
    |---|---|---|---|
    |LLaMA|Meta|65B|[paper](https://arxiv.org/abs/2302.13971v1)|
    |LLaMA2|Meta||[paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)|
    |Alpaca|斯坦福|7B|效果可媲美175B的LLM|
    |vicuna||||
    |Chinese-LLaMA-Alpaca|哈工大||https://github.com/ymcui/Chinese-LLaMA-Alpaca|
    |GLM|清华||https://github.com/THUDM/GLM|
    |ChatGLM-6B|清华||https://github.com/THUDM/ChatGLM-6B|
    |ChatGLM2-6B|清华||https://github.com/THUDM/ChatGLM2-6B|
    |instructGPT|openAI|||
    |GPT-4|openAI|||