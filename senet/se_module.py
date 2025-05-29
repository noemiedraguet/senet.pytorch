from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, threshold = 0.5):
        super(SELayer, self).__init__()
        self.history = []
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
            nn.Threshold(threshold, 0)
        )

    def forward(self, x):
        print(f"SELayer forward called, current history length: {len(self.history)}")
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y_simple = (y.view(b, c))
        y_bool = y_simple == 0
        y_bool = y_bool.tolist()
        coverage_list = []
        for image_channels in y_bool:
            coverage = sum(image_channels)/len(image_channels)
            coverage_list.append(coverage)
        for elem in coverage_list:
            self.history.append(elem)
        return x * y.expand_as(x)
