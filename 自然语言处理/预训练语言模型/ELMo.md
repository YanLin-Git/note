# ELMo

**word2vec**可以对每个单词进行编码，但无法解决多义词问题。例如"bank"，有两个常用含义，但是在**word2vec**中，只对应一行向量。  
**ELMo**的想法是根据上下文，去动态调整 word embedding

- paper: https://arxiv.org/abs/1802.05365v2
- 网络结构
    1. 第一层，word embedding，捕获`单词特征`
    2. 第二层，两个方向的LSTM，捕获`语法特征`
    3. 第三层，两个方向的LSTM，捕获`语义特征`
- 训练过程:
    - 输入: 一批sentence，例如: `<bos>` I love cake `<eos>`
    - 输出: 根据当前词预测下一个词，例如: I love cake `<eos>` `<pad>`
- 参考代码: https://github.com/HIT-SCIR/ELMoForManyLangs/tree/b3de5f1dc1ac13638a930b49c41e1f1e0e185ca1
- 代码实现
1. 第一层，包含 **word_embedding**、 **char_embedding**卷积后的结果

```python
    class ConvTokenEmbedder(nn.Module):
        def __init__(self, config, vocab_size, char_size):
            super(ConvTokenEmbedder, self).__init__()
            self.config = config

            self.word_emb_layer = nn.Embedding(vocab_size, word_embed_dim)
            self.char_emb_layer = nn.Embedding(char_size, char_embed_dim)
            self.activation = nn.ReLU()
            self.output_dim = config['encoder']['projection_dim'] # 第一层最后输出的维度
            self.emb_dim = 0
            if word_emb_layer is not None:
                self.emb_dim += word_embedding_dim

            if char_emb_layer is not None:
                self.convolutions = []
                cnn_config = config['token_embedder']
                filters = cnn_config['filters'] # filters示例: [[1, 32], [2, 32], [3, 64], [4, 128]]
                char_embed_dim = cnn_config['char_dim']

                for i, (width, num) in enumerate(filters):
                    conv = torch.nn.Conv1d(
                        in_channels=char_embed_dim,
                        out_channels=num,
                        kernel_size=width,
                        bias=True
                    )
                    self.convolutions.append(conv)

            self.convolutions = nn.ModuleList(self.convolutions)
            
            self.n_filters = sum(f[1] for f in filters) # 获取卷积核的数量，即卷积层的输出通道
            
            self.n_highway = cnn_config['n_highway']
            self.highways = Highway(self.n_filters, self.n_highway, activation=torch.nn.functional.relu)
            
            self.emb_dim += self.n_filters # 每个单词会有一个 word_embed + char_embed

            self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True) # 最后映射到output_dim维
            
        def forward(self, word_inp, chars_inp, shape):
            '''
            word_inp示例: <bos> I love cake <eos>
            chars_inp示例: <bos> I<pad><pad><pad> love cake <eos> #将每个单词填充到最大长度，方便后面对每个单词做卷积
            word_inp: (batch_size, seq_len)
            chars_inp: (batch_size, seq_len, token_len)
            '''

            embs = []
            batch_size, seq_len = shape
            if self.word_emb_layer is not None:
                batch_size, seq_len = word_inp.size(0), word_inp.size(1)
                word_emb = self.word_emb_layer(word_inp)
                embs.append(word_emb)

            if self.char_emb_layer is not None:
                chars_inp = chars_inp.view(batch_size * seq_len, -1) # (batch_size, seq_len, token_len) --> (batch_size*seq_len, token_len)
            character_embedding = self.char_emb_layer(chars_inp) # (batch_size*seq_len, token_len, char_dim)
            character_embedding = torch.transpose(character_embedding, 1, 2) # (batch_size*seq_len, char_dim, token_len)

            convs = []
            for i in range(len(self.convolutions)):
                convolved = self.convolutions[i](character_embedding) # (batch_size*sequence_length, filter_nums[i], some_len)
                convolved, _ = torch.max(convolved, dim=-1) # (batch_size*sequence_length, filter_nums[i])
                convolved = self.activation(convolved)
                convs.append(convolved)
            char_emb = torch.cat(convs, dim=-1) # (batch_size*sequence_length, filter_nums之和 = self.n_filters)
            char_emb = self.highways(char_emb) # 同上，这个函数会在下边介绍

            embs.append(char_emb.view(batch_size, -1, self.n_filters))
            
            token_embedding = torch.cat(embs, dim=2) # (batch_size, seq_len, word_embedding_dim + n_filters)

            return self.projection(token_embedding) # (batch_size, seq_len, projection_dim)
```
2. 上面用到的**highways**
```python
    class Highway(torch.nn.Module):
        def __init__(self,
                    input_dim: int,
                    num_layers: int = 1) -> None:
            super(Highway, self).__init__()
            self._input_dim = input_dim # 对应class ConvTokenEmbedder中的n_filters
            self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                                for _ in range(num_layers)])
            self._activation = nn.ReLU()
            for layer in self._layers:
                layer.bias[input_dim:].data.fill_(1)

        @overrides
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
            current_input = inputs
            for layer in self._layers:
                projected_input = layer(current_input) # 从 input_dim 映射到 input_dim*2
                linear_part = current_input

                # 将 input_dim*2 拆分为两部分，前半部分直接使用激活函数，后半部分使用sigmoid计算一个gate值
                nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
                gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
                nonlinear_part = self._activation(nonlinear_part)
                gate = torch.sigmoid(gate)

                # 最后的结果为: gate * input + (1-gate) * 激活后的值
                current_input = gate * linear_part + (1 - gate) * nonlinear_part

            return current_input
```

3. 后面两层，两个方向的LSTM:
```python
    class ELMoLstmEncoder(nn.Module):
        '''
        双向LSTM解码器，获取序列每一时刻、每一层的前向、后向表示
        pytorch的lstm可以直接构建多层，但无法得到中间层的hidden表示
        用单层搭建
        '''

        def __init__(self, input_dim, hidden_dim, num_layers):
            super(ELMoLstmEncoder, self).__init__()
            # LSTM各中间层及输出层和输入表示层维度相同
            self.projection_dim = input_dim
            self.num_layers = num_layers

            # 前向LSTM
            self.forward_layers = nn.ModuleList()
            # 前向LSTM投射层：hidden_dim->self.projection_dim
            self.forward_projection = nn.ModuleList()
            # 后向LSTM
            self.backward_layers = nn.ModuleList()
            # 后向投射层
            self.backward_projection = nn.ModuleList()

            lstm_input_dim = input_dim
            for _ in range(num_layers):
                forward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
                forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)
                backward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
                backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

                lstm_input_dim = self.projection_dim

                self.forward_layers.append(forward_layer)
                self.forward_projection.append(forward_projection)
                self.backward_layers.append(backward_layer)
                self.backward_projection.append(backward_projection)

        def forward(self, inputs, lengths):
            '''
            inputs:(batch_size, seq_len, input_dim)
            lengths: (batch_size)每个句子的真实长度
            :return:
            '''
            batch_size, seq_len, input_dim = inputs.shape
            
            # 构建反向lstm的输入
            # [1,2,3,0,0] -> [3,2,1,0,0]
            rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # repeat，对应维度重复对应次
            # rev_idx (batch_size, seq_len)
            for i in range(lengths.shape[0]):
                rev_idx[i, :lengths[i]] = torch.arange(lengths[i] - 1, -1, -1)  # (start, end, step)不含end
            rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
            rev_inputs = inputs.gather(1, rev_idx)

            forward_inputs, backward_inputs = inputs, rev_inputs

            # 保存每一层前后向隐层状态
            stacked_forward_states, stacked_backward_states = [], []
            for layer_index in range(self.num_layers):
                packed_forward_inputs = pack_padded_sequence(
                    forward_inputs, lengths, batch_first=True, enforce_sorted=False
                ) # (batch_size, seq_len, input_dim)
                packed_backward_inputs = pack_padded_sequence(
                    backward_inputs, lengths, batch_first=True, enforce_sorted=False
                ) # (batch_size, seq_len, input_dim)

                forward_layer = self.forward_layers[layer_index]
                packed_forward, _ = forward_layer(packed_forward_inputs) # (batch_size, seq_len, hidden_dim)
                forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
                forward = self.forward_projection[layer_index](forward) # (batch_size, seq_len, projection_dim)
                stacked_forward_states.append(forward)

                backward_layer = self.backward_layers[layer_index]
                packed_backward, _ = backward_layer(packed_backward_inputs)
                backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
                backward = self.backward_projection[layer_index](backward)
                # 恢复至序列的原始顺序
                stacked_backward_states.append(backward.gather(1, rev_idx))

            return stacked_forward_states, stacked_backward_states
```

4. 最后就可以组装起来
```python
    class BiLM(nn.Module):
        def __init__(self, config, vocab_size, char_size):
            super(BiLM, self).__init__()
            self.dropout_prob = config.dropout
            self.num_classes = len(vocab_size)
            # 词表示编码器
            self.token_embedder = ConvTokenEmbedder(
                config,
                vocab_size,
                char_size
            )
            # EMLo LSTM编码器
            self.encoder = ELMoLstmEncoder(
                config.projection_dim,
                config.hidden_dim,
                config.num_layers
            )

            self.classifier = nn.Linear(config.projection_dim, self.num_classes)

        def forward(self, word_inp, chars_inp, shape, lengths):
            '''
            inputs: (batch_size, seq_len, token_len)
            lengths: (batch_size)
            :return:
            '''
            token_embeds = self.token_embedder(word_inp, chars_inp, shape)  # (batch_size, seq_len, projection_dim)
            token_embeds = F.dropout(token_embeds, self.dropout_prob)
            forward, backward = self.encoder(token_embeds, lengths)  # forward (num_layers, batch_size, seq_len, hidden_dim)
            # 取前后向LSTM最后一层的表示计算语言模型输出
            return self.classifier(forward[-1]), self.classifier(backward[-1])
```