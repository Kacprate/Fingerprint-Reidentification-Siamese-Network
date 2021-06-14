import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            # nn.Linear(256 * 4 * 4, 512, bias=True),
            nn.Linear(256 * 6 * 6, 1024, bias=True),
            nn.ReLU(inplace=True),

            # nn.Linear(512, 256, bias=True),
            # nn.ReLU(inplace=True),

            nn.Linear(1024, self.config.latent_size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        # x, indices = torch.max(x, dim=3)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(128, 128))
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 13 * 13, self.config.latent_size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoded_image = self.encoder(x)
        decoded_image = self.decoder(encoded_image)

        latent = encoded_image.view(encoded_image.shape[0], -1)
        latent = self.fc(latent)
        
        return latent, decoded_image

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Linear(self.config.latent_size, 1024, bias=True),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.compare = nn.Sequential(
            nn.Linear(512 * 2, 512, bias=True),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 64, bias=True),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)

        output = self.compare(torch.cat((x1, x2), dim=1))
        return output
