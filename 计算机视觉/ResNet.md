# ResNet

## 一、残差块

- 代码实现
    ```python
    class Residual(nn.Module):
        def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
            super(Residual, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if self.conv3:
                X = self.conv3(X)
            return F.relu(Y + X)
    
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert(in_channels==out_channels)
        
        blk = []
        for i in range(num_residuals):
            if i==0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        
        return nn.Sequential(*blk)
    ```

## 二、ResNet_18

- 代码实现
    ```python 
    class ResNet(nn.Module):
        
        def __init__(self):
            super(ResNet, self).__init__()
        
            self.conv = nn.Sequential()

            # 第一块，与GoogLeNet大致相同
            self.conv.add_module("block", nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),                         # (bs, 3, 224, 224)-->(bs, 64, 112, 112)
                nn.BatchNorm2d(64)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                              # (bs, 64, 112, 112)-->(bs, 64, 56, 56)
            ))

            self.conv.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))  # (bs, 64, 56, 56)-->(bs, 64, 56, 56)
            self.conv.add_module("resnet_block2", resnet_block(64, 128, 2))                   # (bs, 64, 56, 56)-->(bs, 128, 28, 28)
            self.conv.add_module("resnet_block3", resnet_block(128, 256, 2))                  # (bs, 128, 28, 28)-->(bs, 256, 14, 14)
            self.conv.add_module("resnet_block4", resnet_block(256, 512, 2))                  # (bs, 256, 14, 14)-->(bs, 512, 7, 7)

            self.fc = nn.Linear(512, 1000)

        def forward(self, img):
            # img: (batch_size, 3, 224, 224)
            #      (batch_size, channel, height, width)
            feature = self.conv(img)                                                          # (bs, 3, 224, 224)-->(bs, 512, 7, 7)
            feature = F.avg_pool2d(feature, kernel_size=feature.shape[2:])                    # (bs, 512, 7, 7)-->(bs, 512, 1, 1)
            feature = feature.view(img.shape[0], -1)                                          # (bs, 512, 1, 1)-->(bs, 512)
            output = self.fc(feature)                                                         # (bs, 512)-->(bs, 1000)
            return output
    ```