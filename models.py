import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 59 * 59, self.config.latent_size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        # x, indices = torch.max(x, dim=3)
        x = x.view(self.config.batch_size, -1)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Linear(self.config.latent_size * 2, 1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(32, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        output = self.model(torch.cat((x1, x2), dim=1))
        return output
