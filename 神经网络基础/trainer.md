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

## 4、实现自己的Trainer
> 我们也可以包装一下huggingface的Trainer，来实现特定的训练流程
    >> 例如训练一个基于bert的双塔模型  
    >> 只需继承huggingface的Trainer  
    >> 然后实现自己的`compute_loss`，完成自己需要的前向传播、损失计算即可
```python
class MyTrainer(Trainer):
    def compute_loss(self, model: BertModel, inputs: Dict[str, torch.Tensor], return_outputs=False,):

        # 前向传播
        source_sentence_embeddings = model(inputs['text1_input_ids'], inputs['text1_attn_masks'])
        target_sentence_embeddings = model(inputs['text2_input_ids'], inputs['text2_attn_masks'])
        y_pred = nn.functional.cosine_similarity(source_sentence_embeddings, target_sentence_embeddings)

        # 计算损失
        loss = loss_fun(y_pred, inputs['labels'])
        return loss
```

## 5、huggingface中的Trainer更多功能
```python
class Trainer:
    def train():
        self._move_model_to_device(self.model, args.device) # 将模型放置在GPU、TPU上
        self._load_from_checkpoint(resume_from_checkpoint) # 加载之前已经训练的checkpoint
        self._inner_training_loop(): # 真正开始训练
            train_dataloader = self.get_train_dataloader() # 即DataLoader(Dataset, batch_size, collate_fn)
            create_optimizer_and_scheduler() # 创建optimizer、scheduler
            self.model.gradient_checkpointing_enable() # 使模型支持gradient_checkpoint
            model = self._wrap_model(self.model_wrapped) # 使模型支持并行训练
            for epoch in range(num_train_epochs):
                epoch_iterator = train_dataloader
                for step, inputs in enumerate(epoch_iterator):
                    
                    # 核心训练代码
                    tr_loss_step = self.training_step(model, inputs):
                        model.train() # 对应model.eval()
                        inputs = self._prepare_inputs(inputs) # 将inputs转换为tensor，并放置在GPU、TPU上
                        loss = self.compute_loss(model, inputs):
                            outputs = model(**inputs) # 前向传播、计算损失
                        self.accelerator.backward(loss) # 反向传播
                    nn.utils.clip_grad_norm_() # 梯度裁剪
                    self.optimizer.step() # 参数更新
                    self.lr_scheduler.step() # 学习率更新
                    model.zero_grad() # 梯度清零

                    # 达到指定步数时，评估、保存模型
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval):
                        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval): # 根据验证集计算评价指标
                            self.evaluation_loop():
                                model = self._wrap_model(self.model, training=False, dataloader=dataloader)
                                model.eval() # 对应model.train()
                                for step, inputs in enumerate(dataloader):
                                    loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels)) # 计算评价指标

                        self._save_checkpoint(model, trial, metrics=metrics) # 保存模型
```