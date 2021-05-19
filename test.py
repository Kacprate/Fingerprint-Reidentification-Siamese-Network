import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader

from config import Config
from models import Encoder, SiameseNetwork

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

encoder = Encoder(config)
siamese_network = SiameseNetwork(config)

encoder.load_state_dict(torch.load(config.saved_models_folder + '/encoder.pth'))
encoder.to(device)
encoder.eval()

siamese_network.load_state_dict(torch.load(config.saved_models_folder + '/siamese_network.pth'))
siamese_network.to(device)
siamese_network.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(128, 128)),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(config.data_folder, transform=transform)
train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

for batch in train_data_loader:
    batch = batch[0]
    break
batch = batch.to(device)

print('test')
print('the same images')
img1 = batch[0]
features = encoder(img1.unsqueeze(0))
result = siamese_network(features, features)
print(result)
print()

print('different images')
img1, img2 = batch[0], batch[31]
features1, features2 = encoder(img1.unsqueeze(0)), encoder(img2.unsqueeze(0))
result = siamese_network(features1, features2)
print(result)