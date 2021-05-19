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
    transforms.RandomCrop(size=128),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])
train_data = torchvision.datasets.ImageFolder(config.data_folder, transform=transform)
train_data_loader1 = DataLoader(train_data, batch_size=config.batch_size, shuffle=False)
# train_data_loader2 = DataLoader(train_data, batch_size=config.batch_size, shuffle=False)

# transform2 = transforms.Compose([
#     transforms.RandomRotation(degrees=10),
#     transforms.ToTensor()
# ])

loss_function = nn.MSELoss()

epochs = config.epochs
for epoch in range(epochs):
    epoch_loss = 0

    for img_batch in tqdm.tqdm(train_data_loader1):
    # for img_batch, img_batch2 in tqdm.tqdm(zip(train_data_loader1, train_data_loader2)):
        img_batch = img_batch[0]
        img_batch = img_batch.to(device)

        # img_batch2 = img_batch2[0]
        # img_batch2 = img_batch2.to(device)

        # img_batch1 = transform(img_batch)

        optimizer.zero_grad()

        features = encoder(img_batch)
        features2 = encoder(img_batch)
        good_pairs = siamese_network(features, features2)
        bad_pairs = siamese_network(features, features2.roll(shifts=1))

        good_target = torch.ones_like(good_pairs)
        bad_target = torch.zeros_like(bad_pairs)

        output = torch.cat((good_pairs, bad_pairs))
        target = torch.cat((good_target, bad_target))

        # print(output[:10], output[-10:])
        loss = loss_function(output, target)
        # print(loss.item())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(epoch_loss)

torch.save(encoder.state_dict(), config.saved_models_folder + "/encoder.pth")
torch.save(siamese_network.state_dict(), config.saved_models_folder + "/siamese_network.pth")