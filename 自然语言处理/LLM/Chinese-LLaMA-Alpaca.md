# Chinese-LLaMA-Alpaca训练流程
> 项目地址: https://github.com/ymcui/Chinese-LLaMA-Alpaca

## 0、准备工作
> 这里以`llama-7b`、`Chinese-LLaMA-Plus-7B`、`Chinese-Alpaca-Plus-7B`为例
1. `原始llama`模型，可直接下载HF版: `git lfs clone https://huggingface.co/yahma/llama-7b-hf`
2. 通用中文语料，例如这个: [开源书籍](https://github.com/shjwudp/shu/tree/master/books)
    - 官方项目中使用`通用120G`
3. 指令精调数据，例如官方项目中的 `data/alpaca_data_zh_51k.json`
    - 官方项目中使用`指令4M`
4. 24G单卡4090

## 1、扩充词表
1. 在通用中文语料上训练自己的词表
    - 参考这里[sentencepiece](自然语言处理/subword?id=_4-如何使用这几种subword算法？)
2. 也可以直接使用[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)官方训练好的词表`chinese_sp.model`
3. 准备好词表后，使用脚本`merge_tokenizers.py`进行合并
```shell
python scripts/merge_tokenizer/merge_tokenizers.py \
  --llama_tokenizer_dir llama-tokenizer所在目录 \
  --chinese_sp_model_file 用sentencepiece训练的中文词表文件, 例如官方提供的 'chinese_sp.model'文件
```

<details>
<summary><b>最后输出格式</b></summary>

```shell
ls -alh merged_tokenizer_hf merged_tokenizer_sp # 查看这两个目录


merged_tokenizer_hf: # huggingface格式
总用量 760K
drwxr-xr-x  2 root root 4.0K  7月 14 13:15 .
drwxr-xr-x 14 root root 4.0K  7月 17 10:12 ..
-rw-r--r--  1 root root  411  7月 17 09:57 special_tokens_map.json
-rw-r--r--  1 root root  727  7月 17 09:57 tokenizer_config.json
-rw-r--r--  1 root root 741K  7月 17 09:57 tokenizer.model

merged_tokenizer_sp: # 二进制格式
总用量 752K
drwxr-xr-x  2 root root 4.0K  7月 14 13:15 .
drwxr-xr-x 14 root root 4.0K  7月 17 10:12 ..
-rw-r--r--  1 root root 741K  7月 17 09:57 chinese_llama.model
```

</details>

## 2、使用中文语料，对`llama`二次预训练
### 2.1、在`原始llama`基础上预训练
- 修改并运行脚本`run_pt.sh`，需要修改的部分参数如下:
    - `pretrained_model`: `原始llama`模型所在目录
    - `chinese_tokenizer_path`: 扩充后词表所在目录
    - `dataset_dir`: 预训练数据所在目录
    - `data_cache`: 存放数据缓存的目录
    - `output_dir`: 模型权重的输出路径，例如`./models/chinese_llama_lora_test`

<details>
<summary><b>最后输出格式</b></summary>

```shell
ls -alh models/chinese_llama_lora_test models/chinese_llama_lora_test/pt_lora_model


models/chinese_llama_lora_test:
总用量 24K
drwxr-xr-x 3 root root 4.0K  7月 14 15:24 .
drwxr-xr-x 8 root root 4.0K  7月 17 10:03 ..
-rw-r--r-- 1 root root  185  7月 17 09:57 all_results.json
drwxr-xr-x 2 root root 4.0K  7月 14 15:24 pt_lora_model      # lora权重所在目录
-rw-r--r-- 1 root root 1.9K  7月 17 09:57 trainer_state.json # 训练过程中的一些状态信息，例如每个epoch的学习率、loss等
-rw-r--r-- 1 root root  185  7月 17 09:57 train_results.json # 训练完成后的 train metrics，包括训练样本数、耗时等

models/chinese_llama_lora_test/pt_lora_model:
总用量 768K
drwxr-xr-x 2 root root 4.0K  7月 14 15:24 .
drwxr-xr-x 3 root root 4.0K  7月 14 15:24 ..
-rw-r--r-- 1 root root  389  7月 17 09:57 adapter_config.json     # lora配置信息
-rw-r--r-- 1 root root  443  7月 17 09:57 adapter_model.bin       # lora参数
-rw-r--r-- 1 root root  411  7月 17 09:57 special_tokens_map.json # 分词器中的special_tokens
-rw-r--r-- 1 root root  747  7月 17 09:57 tokenizer_config.json   # 分词器配置信息
-rw-r--r-- 1 root root 741K  7月 17 09:57 tokenizer.model         # 分词模型
```

</details>

### 2.2、在`Chinese-LLaMA-Plus-7B`基础上预训练
1. 训练一个单独的lora权重
    - 修改并运行脚本`run_pt.sh`，需要修改的部分参数如下:
        - `pretrained_model`: `原始llama`+`Chinese-LLaMA-Plus-7B`合并后模型 所在目录
        - `chinese_tokenizer_path`: 扩充后词表所在目录
        - `dataset_dir`: 预训练数据所在目录
        - `data_cache`: 存放数据缓存的目录
        - `output_dir`: 模型权重的输出路径
    - 其中的`pretrained_model`参数，涉及`原始llama`+`Chinese-LLaMA-Plus-7B`的合并，后面介绍
    - 其余部分与2.1完全相同

2. *对已有的lora权重`Chinese-LLaMA-Plus-7B`进行更新* (可忽略)
    - 修改并运行脚本`run_pt.sh`，需要修改的部分参数如下:
        - `pretrained_model`: `原始llama`所在目录
        - `peft_path`: `Chinese-LLaMA-Plus-7B`权重所在目录，官方脚本`run_pt.sh`中未提供这个参数
        - `chinese_tokenizer_path`: 扩充后词表所在目录
        - `dataset_dir`: 预训练数据所在目录
        - `data_cache`: 存放数据缓存的目录
        - `output_dir`: 模型权重的输出路径

## 3、使用中文语料，对`llama`进行指令精调
### 3.1、在`Chinese-LLaMA-Plus-7B`基础上，从零开始指令精调
- 修改并运行脚本`run_sft.sh`，需要修改的部分参数如下:
    - `pretrained_model`: `原始llama`+`Chinese-LLaMA-Plus-7B`合并后模型 所在目录
    - `chinese_tokenizer_path`: 扩充后词表所在目录
    - `dataset_dir`: 指令精调数据所在目录
    - `validation_file`: 用作验证集的单个指令精调json文件
    - `output_dir`: 模型权重的输出路径，例如`chinese_alpaca_lora_test`
 
<details>
<summary><b>最后输出格式</b></summary>
        
```shell
ls -alh models/chinese_alpaca_lora_test models/chinese_alpaca_lora_test/sft_lora_model


models/chinese_alpaca_lora_test:
总用量 28K
drwxr-xr-x 3 root root 4.0K  7月 17 10:10 .
drwxr-xr-x 8 root root 4.0K  7月 17 10:03 ..
-rw-r--r-- 1 root root  386  7月 17 10:11 all_results.json   # train_results + eval_results
-rw-r--r-- 1 root root  216  7月 17 10:11 eval_results.json  # 评估完成后的 eval metrics，包括训练样本数、耗时等
drwxr-xr-x 2 root root 4.0K  7月 17 10:10 sft_lora_model     # lora权重所在目录
-rw-r--r-- 1 root root 1.9K  7月 17 10:10 trainer_state.json # 训练过程中的一些状态信息，例如每个epoch的学习率、loss等
-rw-r--r-- 1 root root  190  7月 17 10:10 train_results.json # 训练完成后的 train metrics，包括训练样本数、耗时等

models/chinese_alpaca_lora_test/sft_lora_model:
总用量 772K
drwxr-xr-x 2 root root 4.0K  7月 17 10:10 .
drwxr-xr-x 3 root root 4.0K  7月 17 10:10 ..
-rw-r--r-- 1 root root  403  7月 17 10:10 adapter_config.json     # lora配置信息
-rw-r--r-- 1 root root  443  7月 17 10:10 adapter_model.bin       # lora参数
-rw-r--r-- 1 root root   21  7月 17 10:10 added_tokens.json       # 分词器中新添加的token，例如'[PAD]'
-rw-r--r-- 1 root root  435  7月 17 10:10 special_tokens_map.json # 分词器中的special_tokens
-rw-r--r-- 1 root root  747  7月 17 10:10 tokenizer_config.json   # 分词器配置信息
-rw-r--r-- 1 root root 741K  7月 17 10:10 tokenizer.model         # 分词模型
```

</details>

### 3.2、在`Chinese-Alpaca-Plus-7B`基础上指令精调
1. 训练一个单独的lora权重
    - 修改并运行脚本`run_sft.sh`，需要修改的部分参数如下:
        - `pretrained_model`: `原始llama`+`Chinese-LLaMA-Plus-7B`+`Chinese-Alpaca-Plus-7B`合并后模型 所在目录
        - `chinese_tokenizer_path`: 扩充后词表所在目录
        - `dataset_dir`: 指令精调数据所在目录
        - `validation_file`: 用作验证集的单个指令精调json文件
        - `output_dir`: 模型权重的输出路径

2. 对已有的lora权重`Chinese-Alpaca-Plus-7B`进行更新
    - 修改并运行脚本`run_sft.sh`，需要修改的部分参数如下:
        - `pretrained_model`: `原始llama`+`Chinese-LLaMA-Plus-7B`合并后模型 所在目录
        - `peft_path`: `Chinese-Alpaca-Plus-7B`权重所在目录
        - `chinese_tokenizer_path`: 扩充后词表所在目录
        - `dataset_dir`: 指令精调数据所在目录
        - `validation_file`: 用作验证集的单个指令精调json文件
        - `output_dir`: 模型权重的输出路径


## 4、合并lora权重
- 上面的`pretrained_model`参数，设计到lora权重合并，可使用脚本`merge_llama_with_chinese_lora.py`
```shell
python scripts/merge_llama_with_chinese_lora.py \
    --base_model 某个基础模型，例如原始LLaMA模型 所在目录 \
    --lora_model lora权重所在目录，可以写多个 \
    --output_type 指定输出格式，可为pth或huggingface \
    --output_dir 全量模型权重的输出目录，例如 './models/chinese_llama_lora_merged'
```

<details>
<summary><b>合并脚本的简要流程如下:</b></summary>

```python
base_model = LlamaForCausalLM.from_pretrained(base_model_path)          # 1、加载基础模型
for lora_model_path in lora_model_paths:                                # 遍历每个lora权重目录
    tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)         # 2.1、加载lora权重目录下的分词模型
    base_model.resize_token_embeddings(len(tokenizer))                  # 2.2、扩充基础模型的 embedding，使之与扩充后的词表匹配
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path) # 2.3、使用peft库，生成对应的lora模型
    base_model = lora_model.merge_and_unload()                          # 2.4、合并lora模型

tokenizer.save_pretrained(output_dir)                                   # 3、保存分词器
LlamaForCausalLM.save_pretrained(base_model, output_dir)                # 4、保存模型
```

</details>

<details>
<summary><b>最后输出格式</b></summary>

```shell
ls -alh models/chinese_llama_lora_merged


总用量 13G
drwxr-xr-x 2 root root 4.0K  7月 14 17:24 .
drwxr-xr-x 8 root root 4.0K  7月 17 10:03 ..
-rw-r--r-- 1 root root  591  7月 17 09:58 config.json
-rw-r--r-- 1 root root  132  7月 17 09:58 generation_config.json
-rw-r--r-- 1 root root 9.3G  7月 17 09:58 pytorch_model-00001-of-00002.bin # 模型参数所在文件 1/2
-rw-r--r-- 1 root root 3.6G  7月 17 09:58 pytorch_model-00002-of-00002.bin # 模型参数所在文件 2/2
-rw-r--r-- 1 root root  27K  7月 17 09:59 pytorch_model.bin.index.json     # 模型参数的map表，记录每个参数保存在哪个文件中(上面的1/2、2/2)
-rw-r--r-- 1 root root  411  7月 17 09:58 special_tokens_map.json
-rw-r--r-- 1 root root  747  7月 17 09:58 tokenizer_config.json
-rw-r--r-- 1 root root 741K  7月 17 09:58 tokenizer.model
```

</details>

## 上述训练流程汇总

<details>
<summary><b>示例</b></summary>

```shell
# 1. 扩充词表
python scripts/merge_tokenizer/merge_tokenizers.py \
  --llama_tokenizer_dir './models/llama-7b-hf/' \
  --chinese_sp_model_file './scripts/merge_tokenizer/chinese_sp.model'

# 2. 二次预训练，20G显存
sh ./scripts/training/run_pt.sh

# 3. 合并lora权重，1G显存
python scripts/merge_llama_with_chinese_lora.py \
    --base_model /export/servers/chenyanlin/Chinese-LLaMA-Alpaca/models/llama-7b-hf \
    --lora_model /export/servers/chenyanlin/Chinese-LLaMA-Alpaca/models/chinese_llama_lora_test/pt_lora_model \
    --output_type huggingface \
    --output_dir /export/servers/chenyanlin/Chinese-LLaMA-Alpaca/models/chinese_llama_lora_merged

# 4. 指令精调，17G显存
sh ./scripts/training/run_sft.sh
```

</details>

## 5、本地推理
### 5.0、base_model
- 这里使用`原始llama`+`Chinese-LLaMA-Plus-7B`+`Chinese-Alpaca-Plus-7B`作为最终的全量模型

### 5.1、使用Transformers推理
1. 命令行交互，占用15G显存
```shell
python scripts/inference/inference_hf.py \
    --base_model 全量模型所在目录 \
    --with_prompt 是否将输入与prompt模版进行合并。如果加载Alpaca模型，请务必启用此选项！\
    --interactive 以交互方式启动，以便进行多次单轮问答
```
2. web界面交互
```shell
python scripts/inference/gradio_demo.py --base_model 全量模型所在目录
```

### 5.2、llama.cpp量化部署
- todo 参考[官方文档](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/llama.cpp量化部署)

### 5.3、与LangChain进行集成
1. 检索式问答，占用15G显存
```shell
python scripts/langchain/langchain_qa.py \
  --embedding_path 本地的embedding model所在目录 \
  --model_path 全量模型所在目录 \
  --file_path 待进行检索与提问的文档 \
  --chain_type 可以为refine或stuff
```
2. 摘要生成
```shell
python scripts/langchain/langchain_sum.py \
  --model_path 全量模型所在目录 \
  --file_path 待进行摘要的文档 \
  --chain_type 可以为refine或stuff
```