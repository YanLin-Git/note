# ChatGLM


## 1、ChatGLM2模型结构
- `todo`

## 2、ChatGLM2微调
> 1. 准备数据集，直接在[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)下载
> 2. 拉取脚本、模型
> 3. 24G单卡4090

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

