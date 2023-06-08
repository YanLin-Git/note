# LSTM

## 1. 公式
$$
\begin{aligned}
    输入门: & \quad I_t = \sigma (X_t W_{xi} + H_{t-1} W_{hi} + b_i) \\
    遗忘门: & \quad F_t = \sigma (X_t W_{xf} + H_{t-1} W_{hf} + b_f) \\
    输出门: & \quad O_t = \sigma (X_t W_{xo} + H_{t-1} W_{ho} + b_o) \\
    候选记忆细胞: & \quad \tilde{C}_t = \tanh (X_t W_{xc} + H_{t-1} W_{hc} + b_c) \\
    记忆细胞: & \quad  C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t \\
    隐藏状态: & \quad H_t = O_t \odot \tanh (C_t)
\end{aligned}
$$

## 2. pytorch实现，简单版本:
```python
    def lstm(inputs, state, params):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c] = params
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
            F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
            O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
            C_tilde = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
            C = F * C + I * C_tilde
            H = O * C.tanh()
            outputs.append(H)
        return outputs, (H, C)
```

## 3. pytorch接口
```python
    lstm_layer = nn.LSTM(input_size=, hidden_size=num_hiddens)

    X = torch.rand(num_steps, batch_size, input_size) # (num_steps, batch_size, input_size)
    state = None
    Y, state = lstm_layer(X, state)
    # Y : (num_steps, batch_size, hidden_size)
    # state[0] : (batch_size, hidden_size)
    # state[1] : (batch_size, hidden_size)
```