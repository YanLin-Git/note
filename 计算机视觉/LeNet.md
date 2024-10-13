# LeNet

- 概览

    ||LeNet|
    |---|---|
    |数据集|MNIST|
    |网络结构|2层卷积+2层全连接|
    |卷积窗口|5x5<br>5x5|
    |激活函数|sigmoid|

- 代码实现
    ```python
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            # 卷积层块
            self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 5),      # (bs, 1, 28, 28)-->(bs, 6, 24, 24)
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2),      # (bs, 6, 24, 24)-->(bs, 6, 12, 12)
                nn.Conv2d(6, 16, 5),     # (bs, 6, 12, 12)-->(bs, 16, 8, 8)
                nn.Sigmoid(),
                nn.MaxPool2d(2, 2)       # (bs, 6, 8, 8)-->(bs, 16, 4, 4)
            )
            # 全连接层块
            self.fc = nn.Sequential(
                nn.Linear(16*4*4, 120),  # (bs, 16*4*4)-->(bs, 120)
                nn.Sigmoid(),
                nn.Linear(120, 84),      # (bs, 120)-->(bs, 84)
                nn.Sigmoid(),
                nn.Linear(84, 10)        # (bs, 84)-->(bs, 10)
            )

        def forward(self, img):
            # img: (batch_size, 1, 28, 28)
            #      (batch_size, channel, height, width)
            feature = self.conv(img)
            output = self.fc(feature.view(img.shape[0], -1))
            return output
    ```