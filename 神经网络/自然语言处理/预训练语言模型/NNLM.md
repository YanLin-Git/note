# NNLM
- paper: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
- 根据前n个词，预测下一个词
- 代码实现:
```python
    class NNLM(nn.Module):
        def __init__(self):
            super(NNLM, self).__init__()
            self.emb = nn.Embedding(vocab_size, embedding_dim)
            self.linear_xh = nn.Linear(n_step * embedding_dim, hidden_size, bias=False)
            self.bias_h = nn.Parameter(torch.ones(hidden_size))
            self.tanh = nn.Tanh()
            self.linear_hy = nn.Linear(hidden_size, vocab_size, bias=False)
            self.linear_xy = nn.Linear(n_step * m, vocab_size, bias=False)
            self.bias_y = nn.Parameter(torch.ones(vocab_size))

        def forward(self, X):
            X = self.emb(X) # X : (batch_size, n_step, embedding_dim)
            X = X.view(-1, n_step * embedding_dim) # (batch_size, n_step * embedding_dim)
            h = self.tanh(self.linear_xh(X) + self.bias_h) # (batch_size, hidden_size)
            output = self.linear_xy(X) + self.linear_hy(h) + self.bias_y # (batch_size, vocab_size)
            return output
```