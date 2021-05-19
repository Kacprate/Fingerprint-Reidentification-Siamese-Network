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

encoder.to(device)
encoder.train()

siamese_network.to(device)
siamese_network.train()

params = list(encoder.parameters()) + list(siamese_network.parameters())

optimizer = torch.optim.Adam(params, lr=config.lr, betas=(0.9, 0.999))

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(256, 256)),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(config.data_folder, transform=transform)
train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

loss_function = nn.MSELoss()

epochs = config.epochs
for epoch in range(epochs):
    epoch_loss = 0

    for img_batch in tqdm.tqdm(train_data_loader):
        img_batch = img_batch[0]
        img_batch = img_batch.to(device)

        print(img_batch.shape)

        optimizer.zero_grad()

        features = encoder(img_batch)
        good_pairs = siamese_network(features, features)
        bad_pairs = siamese_network(features, features.roll(shifts=1))

        good_target = torch.ones_like(good_pairs)
        bad_target = torch.zeros_like(bad_pairs)

        output = torch.cat((good_pairs, bad_pairs))
        target = torch.cat((good_target, bad_target))

        loss = loss_function(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(epoch_loss)
