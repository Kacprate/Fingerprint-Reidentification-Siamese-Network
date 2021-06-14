import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader

from config import Config
from models import AutoEncoder, SiameseNetwork

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

autoencoder = AutoEncoder(config)
siamese_network = SiameseNetwork(config)

autoencoder_file = '/autoencoder_epoch125_loss0.9346.pth'
siamese_file = '/siamese_network_epoch125_loss0.9346.pth'

if config.load_model:
    autoencoder.load_state_dict(torch.load(config.saved_models_folder + autoencoder_file))
    siamese_network.load_state_dict(torch.load(config.saved_models_folder + siamese_file))

autoencoder.to(device)
autoencoder.train()

siamese_network.to(device)
siamese_network.train()

params = list(autoencoder.parameters()) + list(siamese_network.parameters())

optimizer = torch.optim.Adam(params, lr=config.lr, betas=(0.9, 0.999))

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    # transforms.RandomCrop(size=128),
    # transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])
train_data = torchvision.datasets.ImageFolder(config.data_folder, transform=transform)
train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

transform2 = transforms.Compose([
    transforms.RandomCrop(size=128),
    transforms.RandomRotation(degrees=10),
])

loss_function = nn.MSELoss()

epochs = config.epochs
for epoch in range(1, epochs + 1):
    epoch_loss_siamese = 0
    epoch_loss_autoencoder = 0
    epoch_loss = 0

    # for batch_no, img_batch in enumerate(train_data_loader1):
    for img_batch in tqdm.tqdm(train_data_loader):
        optimizer.zero_grad()

        img_batch = img_batch[0]
        img_batch2 = img_batch.clone()

        img_batch = img_batch.to(device)
        img_batch = transform2(img_batch)

        img_batch2 = img_batch2.to(device)
        img_batch2 = transform2(img_batch2)

        # assert torch.equal(img_batch, img_batch2), "Batches are not equal"

        features, decoded_img = autoencoder(img_batch)
        features2, decoded_img2 = autoencoder(img_batch2)



        bad_features = features2.roll(shifts=np.random.randint(1, config.batch_size - 1), dims=0)

        good_pairs = siamese_network(features, features2)
        bad_pairs = siamese_network(features, bad_features)

        good_target = torch.ones_like(good_pairs)
        bad_target = torch.zeros_like(bad_pairs)

        output = torch.cat((good_pairs, bad_pairs))
        target = torch.cat((good_target, bad_target))



        # print(output[:10], output[-10:])
        ae_loss = loss_function(img_batch, decoded_img)
        siamese_loss = loss_function(output, target)

        epoch_loss_autoencoder += ae_loss.item()
        epoch_loss_siamese = siamese_loss.item()

        loss = siamese_loss + ae_loss
        # loss = loss_function(img_batch, decoded_img)

        # print(loss.item())
        loss.backward()

        optimizer.step()

    epoch_loss = epoch_loss_autoencoder + epoch_loss_siamese
    print(f'siamese: {epoch_loss_siamese}, autoencoder: {epoch_loss_autoencoder}, all: {epoch_loss}')
    
    intloss = int(epoch_loss * 10000) / 10000
    if epoch % config.save_frequency == 0:
        torch.save(autoencoder.state_dict(), f'{config.saved_models_folder}/autoencoder_epoch{epoch}_loss{intloss}.pth')
        torch.save(siamese_network.state_dict(), f'{config.saved_models_folder}/siamese_network_epoch{epoch}_loss{intloss}.pth')
        print('Saved models, epoch: ' + str(epoch))

for batch in train_data_loader:
    batch = batch[0]
    break
batch = batch.to(device)
batch = transform2(batch)

print('test')
print('the same images')
img1 = batch[0]
features, decoded_img = autoencoder(img1.unsqueeze(0))
result = siamese_network(features, features)
print(result)
print()

print('different images')
img1 = batch[0]
features1, decoded_img1 = autoencoder(img1.unsqueeze(0))
for index in range(1, len(batch)):
    img2 = batch[index]
    features2, decoded_img2 = autoencoder(img2.unsqueeze(0))
    result = siamese_network(features1, features2)
    print(result)
