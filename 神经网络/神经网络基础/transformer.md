# transformer

## 一、参考文章
1. 原理介绍: https://jalammar.github.io/illustrated-transformer/
    - 中文翻译版: https://zhuanlan.zhihu.com/p/54356280
2. 代码实现: http://nlp.seas.harvard.edu/2018/04/03/attention.html 
3. 下面代码实现部分，参考这里: https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb

## 二、代码实现
### 1. 先定义一些常用参数
```python
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_q = d_k = d_v = 64  # dimension of Q, K, V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
```

### 2. 多头注意力机制
```python
    class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()

        def forward(self, Q, K, V, attn_mask):
            '''
            计算 z = softmax(Q K / sqrt(d_k)) V
            Q: (batch_size, n_heads, len_q, d_q)
            K: (batch_size, n_heads, len_k, d_k)
            V: (batch_size, n_heads, len_v, d_v)
            attn_mask: (batch_size, len_q, len_q)
            '''
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores: (batch_size, n_heads, len_q, len_k)
            scores.masked_fill_(attn_mask, -1e9) # 被mask的位置，scores设为无穷小，计算softmax权重时，对应0
            attn = nn.Softmax(dim=-1)(scores) # attn: (batch_size, n_heads, len_q, len_k)
            context = torch.matmul(attn, V) # context: (batch_size, n_heads, len_q, d_v)
            return context, attn
            
    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super(MultiHeadAttention, self).__init__()
            self.W_Q = nn.Linear(d_model, d_q * n_heads)
            self.W_K = nn.Linear(d_model, d_k * n_heads)
            self.W_V = nn.Linear(d_model, d_v * n_heads)
            self.linear = nn.Linear(n_heads * d_v, d_model)
            self.layer_norm = nn.LayerNorm(d_model)

        def forward(self, Q, K, V, attn_mask):
            '''
            q: (batch_size, sequence_len, d_model)
            k: (batch_size, sequence_len, d_model)
            v: (batch_size, sequence_len, d_model)
            '''
            residual, batch_size = Q, Q.size(0)
            '''
            (batch_size, sequence_len, d_model)
                -proj-> (batch_size, sequence_len, nheads*d_q)
                -split-> (batch_size, sequence_len, n_heads, d_q)
                -trans-> (batch_size, n_heads, sequence_len, d_q)
            '''
            q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_q).transpose(1,2)  # q_s: (batch_size, n_heads, len_q, d_q)
            k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: (batch_size, n_heads, len_k, d_k)
            v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: (batch_size, n_heads, len_v, d_v)

            # context: (batch_size, n_heads, len_q, d_v)
            # attn: (batch_size, n_heads, len_q, len_k)
            context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: (batch_size, len_q, n_heads * d_v)
            output = self.linear(context) # (batch_size, len_q, n_heads * d_v) -proj-> (batch_size, len_q, d_model)
            output = output + residual # 残差模块
            return self.layer_norm(output), attn # output: (batch_size, len_q, d_model)
```

### 3. 前馈神经网络
```python
    class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
            super(PoswiseFeedForwardNet, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            self.layer_norm = nn.LayerNorm(d_model)

        def forward(self, inputs):
            residual = inputs # inputs : (batch_size, len_q, d_model)
            output = self.relu(self.conv1(inputs.transpose(1, 2)))
            output = self.conv2(output).transpose(1, 2)
            return self.layer_norm(output + residual)
```

### 4. 然后就可以搭建一个encoder_layer
```python
    class EncoderLayer(nn.Module):
        def __init__(self):
            super(EncoderLayer, self).__init__()
            self.enc_self_attn = MultiHeadAttention()
            self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self, enc_inputs, enc_self_attn_mask):
            enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # Q,K,V 都是 enc_inputs
            enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: (batch_size, len_q, d_model)
            return enc_outputs, attn
```

### 5. 然后完成encoder部分
```python
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.src_emb = nn.Embedding(src_vocab_size, d_model)
            self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        def forward(self, enc_inputs): # enc_inputs : (batch_size, source_len)
            enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
            enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
            enc_self_attns = []
            for layer in self.layers:
                enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
                enc_self_attns.append(enc_self_attn)
            return enc_outputs, enc_self_attns
```

### 6. 然后看看decoder_layer
```python
    def get_attn_subsequent_mask(seq):
        
        '''
        seq: (batch_size, sequence_len)
        生成这样的一个mask矩阵，使得decoder只能看到 前边一部分序列
        [[0., 1., 1., 1.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 0.]]
        '''
        
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte()
        return subsequent_mask
        
    class DecoderLayer(nn.Module):
        def __init__(self):
            super(DecoderLayer, self).__init__()
            self.dec_self_attn = MultiHeadAttention()
            self.dec_enc_attn = MultiHeadAttention() # 相比encoder，多了一个encoder与decoder之间的attention
            self.pos_ffn = PoswiseFeedForwardNet()

        def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
            dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
            dec_outputs = self.pos_ffn(dec_outputs)
            return dec_outputs, dec_self_attn, dec_enc_attn
```

### 7. 完成decoder
```python
    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
            self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
            self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

        def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : (batch_size, target_len)
            dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
            dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
            dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

            dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

            dec_self_attns, dec_enc_attns = [], []
            for layer in self.layers:
                dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)
            return dec_outputs, dec_self_attns, dec_enc_attns   
```

### 8. 最后就可以完成transformer
```python
    class Transformer(nn.Module):
        def __init__(self):
            super(Transformer, self).__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
            self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        def forward(self, enc_inputs, dec_inputs):
            enc_outputs, enc_self_attns = self.encoder(enc_inputs)
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
            dec_logits = self.projection(dec_outputs) # dec_logits : (batch_size, target_len, tgt_vocab_size)
            return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```