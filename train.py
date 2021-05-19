from models import Encoder, SiameseNetwork
import torch
from config import Config
import torchvision.transforms as transforms

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

encoder = Encoder(config, device)
siamese_network = SiameseNetwork(config, device)

encoder.to(device)
encoder.train()

siamese_network.to(device)
siamese_network.train()

params = list(encoder.parameters()) + list(siamese_network.parameters())

optimizer = torch.optim.Adam(params, lr=config.lr, betas=(0.9, 0.999))
optimizer.to(device)


transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomCrop(size=20),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor()
])
train_data = torch.datasets.ImageFolder(config.data_folder, transform=transform)

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()