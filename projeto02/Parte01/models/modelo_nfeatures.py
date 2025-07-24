import torch.nn as nn

class ModeloNFeatures(nn.Module):
    def __init__(self, num_classes=10, nf1=64, nf2=128):
        super(ModeloNFeatures, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, nf1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(nf1, nf2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(nf2 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)