# ChatGLM


## 1、ChatGLM2模型结构
- `todo`

## 2、在120G内存，24G显存上，对ChatGLM2微调
> 1. 准备数据集，直接在[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)下载
> 2. 拉取脚本、模型

### 2.1 微调
直接执行`ptuning`目录下的`train.sh`即可，训练过程中会保存相应的`ptuning权重`

### 2.2 微调前后对比
1. 微调前

    <details>
    <summary><b>测试代码:</b></summary>

    ```python
    model_path = "原始模型所在目录"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model = model.eval()

    # 微调前
    response, history = model.chat(tokenizer, "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞", history=[])
    print(response)
    ```
    </details>

- 输入: `类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞`
- 输出效果: `根据您提供的信息，这件上衣采用牛仔布材质，颜色为白色，风格为简约，图案为刺绣，衣样式为外套，衣款式为破洞。`

2. 微调后

    <details>
    <summary><b>测试代码:</b></summary>

    ```python
    # 1、配置文件中提供'pre_seq_len'参数
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
    # 2、模型加载时，会创建一个随机初始化的'model.transformer.prefix_encoder'
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    # 3、加载P-Tuning权重
    prefix_state_dict = torch.load("/训练时的输出目录/checkpoint-600/pytorch_model.bin")
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    model = model.eval()
    model = model.half().cuda()    # 模型的主要参数，使用float16，猜测这里是为了跟训练阶段保持一致
    model.transformer.prefix_encoder.float()    # ptuning的权重，使用float32

    # 微调后
    response, history = model.chat(tokenizer, "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞", history=[])
    print(response)
    ```
    </details>

- 输入: `类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞`
- 输出效果: `这款牛仔外套采用白底白字的刺绣图案，简约的白色与牛仔的搭配，让整体看起来既大气又随性。胸前的小破洞设计，则使这款外套显得更加时尚，同时也增添了一份俏皮感。`

## 3、在13G内存，16G显存上，对ChatGLM2微调
- 没有资源时，可利用kaggle，熟悉一遍流程
> [官网](https://www.heywhale.com/mw/project/64984a7b72ebe240516ae79c)中介绍，最低只需`6.7G显存`即可进行`P-Tuning v2`微调，于是在kaggle上尝试一遍
> 1. 这里有介绍视频: [Kaggle免费使用GPU攻略](https://www.bilibili.com/video/BV1oa411u7uR/?spm_id_from=333.788&vd_source=6f83dfd98e9e60e0535ab4b15f5c8e50)
> 2. 准备数据集，直接在[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)下载
> 3. 准备脚本，从git拉取[ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)

### 3.1 微调前，查看效果
1. 直接加载`THUDM/chatglm2-6b`模型，kaggle上内存只有`13G`，无法加载
2. 加载量化后的`THUDM/chatglm2-6b-int4`的模型，加载成功，推理阶段报错，暂未解决
3. 因此这里使用`bitsandbytes`库，来加载`8bit量化`模型

<details>
<summary><b>代码如下:</b></summary>

```python
# 1、kaggle中需要先安装这两个包
!pip install --upgrade accelerate
!pip install bitsandbytes


# 2、加载模型
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

model_name_or_path = 'THUDM/chatglm2-6b'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0,
    llm_int8_has_fp16_weight = False,
)
model = AutoModel.from_pretrained(model_name_or_path, 
                                  quantization_config = bnb_config,
                                  trust_remote_code=True)
model = model.eval()

# 3、推理
response, history = model.chat(tokenizer, "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞", history=[])
print(response)

# 最终输出结果:
上衣

材质:牛仔布
颜色:白色
风格:简约
图案:刺绣
衣样式:外套
衣款式:破洞

这是一件简约的刺绣外套,使用牛仔布制成,颜色为白色。衣款式为外套,衣样式为破洞。
```

</details>

### 3.2 微调
1. 在`ptuning/main.py`文件中，模型加载部分，添加`BitsAndBytesConfig`参数

    <details>
    <summary><b>示例代码</b></summary>

    ```python
    # 修改前:
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    # 修改后:
    bnb_config = BitsAndBytesConfig(
            load_in_8bit = True,
            llm_int8_threshold = 6.0,
            llm_int8_has_fp16_weight = False,
        )
    model = AutoModel.from_pretrained(model_args.model_name_or_path, 
                                    quantization_config = bnb_config,
                                    config=config,
                                    trust_remote_code=True)
    ```
    
    </details>
2. 将修改好的`ptuning`目录打包: `tar -czvf ptuning.tar.gz ChatGLM2项目路径/ptuning`
3. 将下载好的数据集、打包好的脚本上传至kaggle
4. 在kaggle中执行脚本即可
   
    <details>
    <summary><b>脚本如下:</b></summary>

    ```python
    !pip install rouge_chinese
    !pip install --upgrade accelerate
    !pip install bitsandbytes

    import os
    os.environ["WANDB_DISABLED"] = "true"

    !torchrun --standalone --nnodes=1 --nproc-per-node=1 /kaggle/input/ptuning/ptuning/main.py \
        --do_train \
        --train_file /kaggle/input/advertisegen/AdvertiseGen/train.json \
        --validation_file /kaggle/input/advertisegen/AdvertiseGen/dev.json \
        --preprocessing_num_workers 2 \
        --prompt_column content \
        --response_column summary \
        --overwrite_cache \
        --model_name_or_path THUDM/chatglm2-6b \
        --output_dir output/adgen-chatglm2-6b-pt \
        --overwrite_output_dir \
        --max_source_length 64 \
        --max_target_length 128 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 6 \
        --predict_with_generate \
        --max_steps 600 \
        --logging_steps 60 \
        --save_steps 60 \
        --learning_rate 2e-2 \
        --pre_seq_len 128 
    ```
    
    </details>

### 3.3 微调后，效果对比

<details>
    <summary><b>示例代码:</b></summary>

    ```python
    !pip install --upgrade accelerate
    !pip install bitsandbytes

    import torch
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_name_or_path = 'THUDM/chatglm2-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name_or_path,
                                            trust_remote_code=True,
                                            pre_seq_len=128) # 这里要提供'pre_seq_len'参数
    bnb_config = BitsAndBytesConfig(
                    load_in_8bit = True,
                    llm_int8_threshold = 6.0,
                    llm_int8_has_fp16_weight = False,
                )
    model = AutoModel.from_pretrained(model_name_or_path, 
                                        quantization_config = bnb_config,
                                        config=config,
                                        trust_remote_code=True)


    # 加载P-Tuning权重
    prefix_state_dict = torch.load("/kaggle/input/ptuning-model/pytorch_model.bin")
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model.transformer.prefix_encoder.float()    # ptuning的权重，使用float32
    model.eval()

    # 微调后
    response, history = model.chat(tokenizer, "类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞", history=[])
    print(response)
    ```
    
</details>

- 输入: `类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞`
- 输出效果: `这款牛仔外套采用白色牛仔面料，破洞设计，简约大方。刺绣的图案点缀其中，增添了不少的活力。整体风格简约大方，穿起来十分舒适。`
