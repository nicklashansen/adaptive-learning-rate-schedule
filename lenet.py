import torch.nn as nn


class LeNet5(nn.Module):
    """
    Implementation of LeNet-5 as described in http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf.
    """
    def __init__(self, num_channels_in=3, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels_in, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)
