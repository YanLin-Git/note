# GRU

## 1. 公式
$$
\begin{aligned}
    重置门: & \quad R_t = \sigma (X_t W_{xr} + H_{t-1} W_{hr} + b_r) \\
    更新门: & \quad Z_t = \sigma (X_t W_{xz} + H_{t-1} W_{hz} + b_z) \\
    候选隐藏状态: & \quad \tilde{H}_t = \tanh (X_t W_{xh} + (R_t \odot H_{t-1}) W_{hh} + b_h) \\
    隐藏状态: & \quad  H_t = Z_t \odot H_{t-1} + (1-Z_t) \odot \tilde{H}_t
\end{aligned}
$$

## 2. pytorch实现，简单版本:
```python
    def gru(inputs, state, params):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h = params
        
        '''
        inputs为 num_steps个形状为(batch_size, input_size)的矩阵
        outpus为 num_steps个形状为(batch_size, hidden_size)的矩阵，对应每个时间步的隐藏状态

        X : (batch_size, input_size)
        W_xz、W_xr、W_xh: (input_size, hidden_size)
        X · W_xh: (batch_size, hidden_size)
        W_hz、W_hr、W_hh: (hidden_size, hidden_size)
        H · W_hh: (hidden_size)
        '''

        H, = state
        outputs = []
        for X in inputs:
            Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
            R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
            H_tilde = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            outputs.append(H)
        return outputs, (H,)

```

## 3. pytorch接口
```python
    gru_layer = nn.GRU(input_size=, hidden_size=num_hiddens)

    # 与RNN完全相同
    X = torch.rand(num_steps, batch_size, input_size) # (num_steps, batch_size, input_size)
    state = None
    Y, state = gru_layer(X, state)
    # Y : (num_steps, batch_size, hidden_size)
    # state[0] : (batch_size, hidden_size), state是个元组，这里只用到state[0]
```