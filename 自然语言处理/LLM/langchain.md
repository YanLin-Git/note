# langchain
> 最近刚了解清华推出的[LangChain+ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)，可用来做智能问答，学习一下，整理了个人笔记

## 1. 流程图
![langchain+chatglm.png](jpgs/langchain+chatglm.png) 
- 流程简介:
    1. 构造本地知识库，对应上图中的`1-2-3-4-5-6`
    2. 根据用户的query，查询知识库，对应上图中的`8-9-10-11-12`
    3. 根据查询结果，构造相应的prompt，对应上图中的`12-13`
    4. 输入到LLM，得到最终结果，对应上图中的`14-15`
## 2. 可探讨的优化方向
1. 对于长文本，如何进行拆分？
    - 对应流程图中的`3-4-5`，目前项目中提供了两种方式:
        1. 基于规则
            1. 先按照【。！？】等拆分
            2. 如果单句过长，再按照【，,】等拆分
            3. 仍然超出最大长度，直接按最大长度去拆分
        2. 基于模型
            达摩院开源的 nlp_bert_document-seqmentation_chinese-base
2. 知识库中有，但未检索到
    - 对应流程图中的`5-6`、`8-9`，需要对**embedding模型**做微调，或者尝试其他**embedding模型**。目前支持这几种:
    ```python
    embedding_model_dict = {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec": "GanymedeNil/text2vec-large-chinese",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
    }
    ```
3. 检索到大部分知识不相关
    - 对应流程图中的`10-11`，可以调整相关度阈值，建议是 <500，过滤掉那些不相关的内容
    - 另外其他向量库也值得尝试
        1. 目前langchian-ChatGLM中使用的**FAISS**
        2. 视频介绍中提到接下来会去探索**Chroma**、**Milvus**，更多向量库请参考[langchain_vectorstores](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html)
        
4. 已经检索到相关知识，但回答质量差
    1. 对应流程图中的`12-13`，需要调整Prompt-template，例如官方代码中的两个示例:
    ```python
    query = 'Langchain 能够接入哪些数据类型?'
    docs = vector_sotre.similarity_search(query) # 查询知识库
    context = [doc.page_content for doc in docs]
    context = '\n'.join(context)

    # 第一种模板
    PROMPT_TEMPLATE = '''
    已知信息: {context}

    根据已知信息回答问题: {query}
    '''

    # 第二种模板
    PROMPT_TEMPLATE = '''
    已知信息：{context} 

    根据上述已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 
    问题是：{query}
    '''
    ```
    2. 对应流程图中的`14-15`，需要对LLM进行微调
        - 例如官方代码中采用**ChatGLM-6B**，可以参考微调的介绍: [ChatGLM-6B-PT](https://www.heywhale.com/mw/project/6436d82948f7da1fee2be59e)

## 3. 参考链接
1. [langchain-ChatGLM官方教程](https://www.bilibili.com/video/BV13M4y1e7cN/?spm_id_from=333.337.search-card.all.click&vd_source=6f83dfd98e9e60e0535ab4b15f5c8e50)
2. [langchain-ChatGLM项目地址](https://github.com/imClumsyPanda/langchain-ChatGLM)
3. [ChatGLM-6B项目地址](https://github.com/THUDM/ChatGLM-6B)
4. [langchain官网](https://python.langchain.com/en/latest/index.html)
