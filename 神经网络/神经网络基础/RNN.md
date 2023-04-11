# RNN
## 1. 公式
$$
    H_t = \tanh (X_t \dot W_{xh} + H_{t-1} \dot W_{hh} + b_h)
$$

## 2. pytorch实现，简单版本:
```python
    def rnn(inputs, state, params):
        W_xh, W_hh, b_h = params
        '''
        inputs为 num_steps个形状为(batch_size, input_size)的矩阵
        outpus为 num_steps个形状为(batch_size, hidden_size)的矩阵，对应每个时间步的隐藏状态

        X : (batch_size, input_size)
        W_xh: (input_size, hidden_size)
        X · W_xh: (batch_size, hidden_size)
        W_hh: (hidden_size, hidden_size)
        H · W_hh: (hidden_size)
        '''
        H, = state
        outputs = []
        for X in inputs:
            H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
            # Y = torch.matmul(H, W_hq) + b_q
            outputs.append(H)
        return outputs, (H,)
```

## 3. pytorch接口
```python
    rnn_layer = nn.RNN(input_size=, hidden_size=num_hiddens)

    X = torch.rand(num_steps, batch_size, input_size) # (num_steps, batch_size, input_size)
    state = None
    Y, state = rnn_layer(X, state)
    # Y : (num_steps, batch_size, hidden_size)
    # state[0] : (batch_size, hidden_size), state是个元组，这里只用到state[0]
```