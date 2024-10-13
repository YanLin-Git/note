# AlexNet

- 概览
    > LeNet-->AlexNet

    ||LeNet|AlexNet|备注|
    |---|---|---|---|
    |数据集|MNIST|ImageNet||
    |网络结构|2层卷积+2层全连接|5层卷积+3层全连接||
    |卷积窗口|5x5<br>5x5|`11x11`<br>5x5<br>3x3<br>3x3<br>3x3|ImageNet中图像高、宽都增大<br>需要更大的窗口来捕获物体<br>所以这里第一层卷积用`11x11`|
    |激活函数|sigmoid|ReLU||
    |dropout|否|是||

- 代码实现
    ```python
    class AlexNet(nn.Module):
        def __init__(self):
            super(AlexNet, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 96, 11, 4),       # (bs, 3, 224, 224)-->(bs, 96, 54, 54)
                nn.ReLU(),
                nn.MaxPool2d(3, 2),            # (bs, 96, 54, 54)-->(bs, 96, 26, 26)
                nn.Conv2d(96, 256, 5, 1, 2),   # (bs, 96, 26, 26)-->(bs, 256, 26, 26)
                nn.ReLU(),
                nn.MaxPool2d(3, 2),            # (bs, 256, 26, 26)-->(bs, 256, 12, 12)
                nn.Conv2d(256, 384, 3, 1, 1),  # (bs, 256, 12, 12)-->(bs, 384, 12, 12)
                nn.ReLU(),
                nn.Conv2d(384, 384, 3, 1, 1),  # (bs, 384, 12, 12)-->(bs, 384, 12, 12)
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, 1, 1),  # (bs, 384, 12, 12)-->(bs, 256, 12, 12)
                nn.ReLU(),
                nn.MaxPool2d(3, 2)             # (bs, 256, 12, 12)-->(bs, 256, 5, 5)
            )
            self.fc = nn.Sequential(
                nn.Linear(256*5*5, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 1000),
            )

        def forward(self, img):
            # img: (batch_size, 3, 224, 224)
            #      (batch_size, channel, height, width)
            feature = self.conv(img)
            output = self.fc(feature.view(img.shape[0], -1))
            return output
    ```