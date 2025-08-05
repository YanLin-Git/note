# ZeRO

> paper: [ZeRO](https://arxiv.org/abs/1910.02054v3)


## 一、理论介绍
> 参考: https://zhuanlan.zhihu.com/p/618865052


## 二、deepspeed
> 使用教程: https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration  
> 参考: https://zhuanlan.zhihu.com/p/630734624

<details>
<summary>在代码中使用</summary>

```diff
    # 解析参数
+   parser = argparse.ArgumentParser()
+   parser.add_argument('--deepspeed', type=str)     # deepspeed的配置文件
+   parser.add_argument('--local_rank', default=0, type=int) # deepspeed命令启动时，会指定local_rank参数
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