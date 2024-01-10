# Milvus

# 一、Milvus Lite
> Milvus的轻量版，只适合在测试环境使用  
> 参考文档: https://www.milvus-io.com/milvus_lite

## 1. 安装
- `pip install milvus`
## 2. 配置
```python
from milvus import default_server

with default_server:
    default_server.config.set('system_Log_level', 'info')
    default_server.config.set('proxy_port', 19531)
    default_server.config.set('dataCoord.segment.maxSize', 1024)
    default_server.set_base_dir('milvus_data') # 指定本地目录，存储相关数据
```

# 二、PyMilvus (Milvus Python SDK)

## 1. 安装
- `pip install pymilvus`

## 2. 连接到服务器
```python
from milvus import default_server
from pymilvus import connections

with default_server:
    # Milvus Lite has already started, use default_server here.
    connections.connect(host='localhost', port=default_server.listen_port)
```

## 3. 创建集合
```python
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

# 主键
book_name = FieldSchema(
  name="book_name",
  dtype=DataType.VARCHAR,
  max_length=200,
)
# 向量
book_emb = FieldSchema(
  name="book_emb",
  dtype=DataType.FLOAT_VECTOR,
  dim=2
)
schema = CollectionSchema(
  fields=[book_name, book_emb],
  description="Test book search"
)
collection_name = "book"

collection = Collection(
    name=collection_name,
    schema=schema,
    )
```

## 4. 插入数据
```python
from pymilvus import Collection

# 数据示例
data = [
    ["sanguo", 'shuihu'], # book_name
    [[0.2, 0,5], [0.9, 0.1]] # book_emb
]

# 插入数据
collection = Collection("book") # 上一步创建好的集合
mr = collection.insert(data)

# 插入数据后调用
# 数据可能还在缓存区，需要手动落盘
collection.flush()
```

## 5. 为向量建索引
```python
from pymilvus import Collection, utility

index_params = {
  "metric_type":"IP", # 相似度指标: L2(欧几里得距离)、IP(内积)
  "index_type":"IVF_FLAT", # IVF_FLAT索引，会进行聚类。查询时，先选取最近的n个簇，然后簇内查询
  "params":{"nlist":1024} # 聚类，构建1024个簇
}

collection = Collection("book")      
collection.create_index(
  field_name="book_emb", 
  index_params=index_params
)
```

## 6. 进行向量搜索
```python
from pymilvus import Collection

# 将集合加载到内存
collection = Collection("book")
collection.load()

# 参数配置
search_params = {
    "metric_type": "IP", # 计算距离的方式
    "params": {"nprobe": 10} # 建索引时构建了N个簇，这里指定从最近的10个簇中检索
}

# 搜索topk
results = collection.search(
	data=[[0.1, 0.2]], # 要搜索的向量
	anns_field="book_emb", # 要搜索的字段
	param=search_params,
	limit=10, # topk
	expr=None,
	output_fields=['book_name'] # 返回的字段
	consistency_level="Strong"
)

# 释放集合，减少内存消耗
collection.release()
```