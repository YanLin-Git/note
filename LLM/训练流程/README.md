# 训练流程


1. 训练阶段
    - [instructGPT](LLM/训练流程/instructGPT.md)介绍了LLM训练的3个阶段
        1. 预训练
        2. SFT
        3. RLHF
2. 训练方式
    - 根据自己手头的显卡资源，可以选择不同的训练方式
        1. 全参数微调
        2. 部分参数微调
        3. lora
        4. QLoRA
        5. P-Tuning
    - 几个实例
        ||full|P-Tuning|lora|
        |---|---|---|---|
        |预训练|||2. [Chinese-LLaMA-Alpaca](LLM/训练流程/Chinese-LLaMA-Alpaca.md)|
        |SFT||1. [ChatGLM](LLM/训练流程/ChatGLM.md)|2. [Chinese-LLaMA-Alpaca](LLM/训练流程/Chinese-LLaMA-Alpaca.md)|
        |RLHF||||
3. 训练算法
    1. [PPO](LLM/训练流程/PPO.md)
    2. [DPO](LLM/训练流程/DPO.md)