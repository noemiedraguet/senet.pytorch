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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y_simple = (y.view(b, c))
        y_bool = y_simple == 0
        y_bool = y_bool.tolist()
        print(len(y_bool))
        for image in y_bool:
            image = sum(image)
        
        return x * y.expand_as(x)
