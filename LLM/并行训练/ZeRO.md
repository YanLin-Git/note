# ZeRO

> paper: [ZeRO](https://arxiv.org/abs/1910.02054v3)


## 一、理论介绍
> 参考: https://zhuanlan.zhihu.com/p/618865052

### 1.1 DDP流程回顾
1. 每块GPU上加载一个完整的模型
2. 假设我们有3卡，将一个batch的数据分成3份，每块GPU拿到一份数据后，做自己的forward、backward，并计算自己的梯度
3. 几个GPU之间，对梯度做一次**all-reduce**(`reduce-scatter`+`all-gather`)
4. 每个GPU上执行自己的参数更新

- 整个流程中，每块GPU都保存着完整的`模型参数`、`梯度`、`优化器状态`

<details>
<summary>示意图</summary>

![ddp.png](../jpgs/ddp.png)

</details>

### 1.2 ZeRO中的改进
> 为了进一步降低显存占用，将`模型参数`、`梯度`、`优化器状态`都拆分为多份，每块GPU上仅保存自己需要的那份。

- 每块GPU中保存的内容:
    |stage|模型参数|梯度|优化器状态|
    |---|---|---|---|
    |stage1|完整|完整|部分|
    |stage2|完整|部分|部分|
    |stage3|部分|部分|部分|

#### 1.2.1 stage1
1. 每块GPU上加载一个完整的模型
2. 假设我们有3卡，将一个batch的数据分成3份，每块GPU拿到一份数据后，做自己的forward、backward，并计算自己的梯度
3. 几个GPU之间，对梯度做一次`reduce-scatter`，每个GPU上能获取到一部分`完整梯度`，对应下图中梯度的绿色部分
    - 梯度的白色部分用不到，我们不用关心。
4. 对相应的W部分进行参数更新
5. 对W做一次`all-gather`

<details>
<summary>stage1示意图</summary>

![ZeRO_1.png](../jpgs/ZeRO_1.png)

</details>

#### 1.2.2 stage2

1. 每块GPU上加载一个完整的模型
2. 假设我们有3卡，将一个batch的数据分成3份，每块GPU拿到一份数据后，做自己的forward、backward，并计算自己的梯度
3. 几个GPU之间，对梯度做一次`reduce-scatter`，每个GPU上能获取到一部分`完整梯度`，对应下图中梯度的绿色部分
    - 梯度的白色部分不再使用，释放掉。这里是与stage1的唯一区别
4. 对相应的W部分进行参数更新
5. 对W做一次`all-gather`


<details>
<summary>stage2示意图</summary>

![ZeRO_2.png](../jpgs/ZeRO_2.png)

</details>

#### 1.2.3 stage3
> 每块GPU上不再加载模型所有的参数，仅保存部分参数，只在必要的时候才去读取并完成相应工作

1. 每块GPU上保存部分参数
2. 假设我们有3卡，将一个batch的数据分成3份，每块GPU拿到一份数据
3. 对W做一次`all-gather`，拿到完整参数后，再forward，然后将不需要维护的那部分W释放
4. 对W做一次`all-gather`，拿到完整参数后，再backward，然后将不需要维护的那部分W释放
5. 对梯度做一次`reduce-scatter`，每个GPU上能获取到一部分`完整梯度`，对应下图中梯度的绿色部分
    - 梯度的白色部分不再使用，释放掉。
6. 对自己维护的这部分W进行参数更新

<details>
<summary>stage3示意图</summary>

![ZeRO_3.png](../jpgs/ZeRO_3.png)

</details>

## 二、deepspeed
> 使用教程: https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration  
> 参考: https://zhuanlan.zhihu.com/p/630734624

<details>
<summary>在代码中使用</summary>

```diff
    # 解析参数
+   parser = argparse.ArgumentParser()
+   parser.add_argument('--deepspeed', type=str)     # deepspeed的配置文件
+   args = parser.parse_args()

    # 解析配置文件
+   from transformers.deepspeed import HfTrainerDeepSpeedConfig
+   hf_deepspeed_config = HfTrainerDeepSpeedConfig(args.deepspeed)
    
    # Accelerate DeepSpeed Plugin
+   from accelerate.utils import DeepSpeedPlugin
+   os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
+   deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_deepspeed_config)

    # 初始化accelerator
+   from accelerate import Accelerator
+   accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin) 


    model = MyModel()  # 初始化model
-   # 这里不需要再迁移
-   model = model.to(device)  # model迁移到GPU

    # 如果配置文件中未指定optimizer、scheduler，这里不需要任何改动
    optimizer = torch.optim.Adam(model.parameters())  # 初始化optimizer
    # 如果配置文件中指定了optimizer、scheduler，则使用这种方式创建
+   # from accelerate.utils import DummyOptim, DummyScheduler
+   # optimizer = DummyOptim(params=model.parameters())
+   # lr_scheduler = DummyScheduler(optimizer)

    dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # 将model、optimizer、dataloader放在相应设备上
+   model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(num_epochs):
        for x, y_true in dataloader:
            y_pred = model(x)        # 正向传播
            l = loss(y_pred, true)   # 计算损失函数            
-           l.backward()             # 反向传播，计算梯度
+           accelerator.backward(loss)
            optimizer.step()         # 更新参数
            optimizer.zero_grad()    # 最后这里记得要将梯度清零
```

</details>

1. 启动训练
    - 单机单卡
        1. `deepspeed --num_gpus=1 train.py --deepspeed ds_config.json`
        2. 指定GPU: `deepspeed --include localhost:1 train.py --deepspeed ds_config.json`
    - 单机多卡
        1. `torchrun --nproc_per_node=2 train.py --deepspeed ds_config.json`
        2. `deepspeed --num_gpus=2 train.py --deepspeed ds_config.json`
2. ds_config.json
    <details>
    <summary><b>ZeRO-2 Config</b></summary>

    ```json
    {
        "fp16": {
            "enabled": "auto", // true
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 500
            }
        },

        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            }, // 将部分数据(优化器参数、梯度) offload到CPU，降低对显存的需求
            "allgather_partitions": true,
            "allgather_bucket_size": 2e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true
        },

        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": false
    }

    ```

    </details>