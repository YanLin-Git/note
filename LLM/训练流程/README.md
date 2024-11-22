# 训练流程


1. 训练阶段
    - LLM训练的3个阶段：`预训练`、`指令微调`、`偏好数据训练`
    - [instructGPT](LLM/训练流程/instructGPT.md)介绍了他们`指令微调`、`偏好数据训练`两个阶段的做法

2. 训练方式
    - 根据模型大小、手里的显卡资源，可以选择不同的训练方式：`全参数微调`、`P-Tuning`、`lora`、`QLoRA`等
    - 几个实例

        ||full|P-Tuning|lora|
        |---|---|---|---|
        |预训练|||2. [Chinese-LLaMA-Alpaca](LLM/训练流程/Chinese-LLaMA-Alpaca.md)|
        |指令微调||1. [ChatGLM](LLM/训练流程/ChatGLM.md)|2. [Chinese-LLaMA-Alpaca](LLM/训练流程/Chinese-LLaMA-Alpaca.md)|
        |偏好数据训练||||

3. 训练算法
    - 目前在整理`偏好数据训练`阶段的几个算法
        1. [PPO](LLM/训练流程/PPO.md)
        2. [DPO](LLM/训练流程/DPO.md)