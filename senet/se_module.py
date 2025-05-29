from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, threshold = 0.5):
        super(SELayer, self).__init__()
        #The history attribute of each layer contains the average number of channels switched off in this layer for each image of the dataset
        self.history = []
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
            #Adding a threshold to the SE Layer
            nn.Threshold(threshold, 0)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #y_simple contains a list where each element is relative to one image of the batch. Each element is a list of descriptors for each channel of the SELayer.
        y_simple = (y.view(b, c))
        #y_bool replaces the values of y_simple with boolean values (true if the descriptor is 0, false otherwise).
        y_bool = y_simple == 0
        y_bool = y_bool.tolist()
        coverage_list = []
        #For each element of y_bool (relative to one image), the coverage is the percentage of channels put to 0 for this image in this layer.
        for image_channels in y_bool:
            coverage = sum(image_channels)/len(image_channels)
            coverage_list.append(coverage)
        #The percentage of switched off channels for each image is added to the history of the layer, to be able to get an average across the whole dataset later.
        for elem in coverage_list:
            self.history.append(elem)
        return x * y.expand_as(x)
