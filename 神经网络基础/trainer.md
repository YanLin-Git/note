# 训练流程

## 1、基本的训练流程
> 一个基本的训练流程需要包含下面这几步:
```python
for epoch in range(num_epochs):
    for x, y_true in train_iter:
        y_pred = net(x)          # 正向传播
        l = loss(y_pred, true)   # 计算损失函数            
        l.backward()             # 反向传播，计算梯度
        optimizer.step()         # 更新参数
        optimizer.zero_grad()    # 最后这里记得要将梯度清零
```

## 2、添加读取数据的流程
> 使用`torch.utils.data`中的Dataset、DataLoader:
```python
for epoch in range(num_epochs):
    for x, y_true in DataLoader(Dataset, batch_size=batch_size, collate_fn=collate_data):
        y_pred = net(x)          # 正向传播
        l = loss(y_pred, true)   # 计算损失函数            
        l.backward()             # 反向传播，计算梯度
        optimizer.step()         # 更新参数
        optimizer.zero_grad()    # 最后这里记得要将梯度清零
```

## 3、huggingface中的Trainer
> 了解前面的基本流程后，来看看huggingface中Trainer的一些主要操作
```python
class Trainer:
    def train():
        # pre: 一些准备工作
        self._inner_training_loop(): # 真正开始训练
            train_dataloader = self.get_train_dataloader() # 即DataLoader(Dataset, batch_size, collate_fn)
            for epoch in range(num_train_epochs):
                epoch_iterator = train_dataloader
                for step, inputs in enumerate(epoch_iterator):
                    tr_loss_step = self.training_step(model, inputs):
                        loss = self.compute_loss(model, inputs):
                            outputs = model(**inputs) # 前向传播、计算损失
                        self.accelerator.backward(loss) # 反向传播
                    nn.utils.clip_grad_norm_() # 梯度裁剪
                    self.optimizer.step() # 参数更新
                    self.lr_scheduler.step() # 学习率更新
                    model.zero_grad() # 梯度清零
```

## 4. 实现自己的Trainer
> 我们也可以包装一下huggingface的Trainer，来实现特定的训练流程
    >> 例如训练一个基于bert的双塔模型  
    >> 只需继承huggingface的Trainer  
    >> 然后实现自己的`compute_loss`，完成自己需要的前向传播、损失计算即可
```python
class MyTrainer(Trainer):
    def compute_loss(
        self,
        model: BertModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs=False,
    ):

    # 前向传播
    source_sentence_embeddings = model(inputs['text1_input_ids'], inputs['text1_attn_masks'])
    target_sentence_embeddings = model(inputs['text2_input_ids'], inputs['text2_attn_masks'])
    y_pred = nn.functional.cosine_similarity(source_sentence_embeddings, target_sentence_embeddings)

    # 计算损失
    loss = loss_fun(y_pred, inputs['labels'])
    return loss
```