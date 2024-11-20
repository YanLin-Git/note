# peft

> 项目地址: https://github.com/huggingface/peft

## 1、peft_model的两种加载方式:
1. 已经有base_model、peft_model
    - 获取peft_model所在目录，执行: `model = PeftModel.from_pretrained(base_model, peft_model_path)`
2. 只有base_model，训练一个全新的peft_model
    - 需要自己指定一些配置信息，执行: `model = get_peft_model(base_model, peft_config)`

## 2、几种训练方式:
1. `lora`
2. *Prompt-Tuning*
3. *P-Tuning*
4. `Prefix-Tuning` (P-Tuning v2)

### 2.1 lora
- 模型中的`nn.Linear`或者`nn.Conv1d`，都可以增加一个lora层

<details>
<summary><b>主要代码如下:</b></summary>

```python
# 初始化
# r、lora_alpha、lora_dropout 为配置文件中指定的参数
self.lora_A = nn.Linear(self.in_features, r, bias=False)
self.lora_B = nn.Linear(r, self.out_features, bias=False)
self.scaling = lora_alpha / r
self.lora_dropout = nn.Dropout(p=lora_dropout)


# forward
result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias) # 先拿到‘nn.Linear’或者‘nn.Conv1d’的结果

result += (
    self.lora_B(
        self.lora_A(self.lora_dropout(x))
    )
    * self.scaling
) # 核心计算公式
```

</details>

### 2.4 Prefix-Tuning
- 通过HuggingFace框架内置的past_key_values参数控制
    - 在`attention_fn`函数中，key、value前先拼接一些token_embedding，再进行后续计算
    
<details>
<summary><b>主要代码如下:</b></summary>

```python
# 初始化，transformers中每层的key、value都需要添加，因此维度是: num_layers * hidden_size * 2
self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_layers * config.hidden_size * 2)
past_key_values = self.embedding(prefix)


# 'past_key_values'拆分，变形之后，传递到每层的'layer_past'，为两个tensor组成的数组
# 每个tensor形状为 (pre_seq_len, batch_size, hidden_size)
# forward时，看起来就很简单:
past_key, past_value = layer_past[0], layer_past[1]       # past_key_values参数传入的内容，这里就是虚拟token对应的embedding
key_layer = torch.cat((past_key, key_layer), dim=0)       # 拼接在key_layer最前面
value_layer = torch.cat((past_value, value_layer), dim=0) # 拼接在value_layer最前面
```

</details>
    