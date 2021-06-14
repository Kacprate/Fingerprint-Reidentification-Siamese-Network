import itertools

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader

from config import Config
from models import AutoEncoder, SiameseNetwork

batch_size = 32
threshold = 0.5

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

autoencoder = AutoEncoder(config)
siamese_network = SiameseNetwork(config)

autoencoder_file = '/autoencoder_epoch125_loss0.9346.pth'
siamese_file = '/siamese_network_epoch125_loss0.9346.pth'

autoencoder.load_state_dict(torch.load(config.saved_models_folder + autoencoder_file))
autoencoder.to(device)
autoencoder.eval()

siamese_network.load_state_dict(torch.load(config.saved_models_folder + siamese_file))
siamese_network.to(device)
siamese_network.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(config.data_folder, transform=transform)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

transform2 = transforms.Compose([
    transforms.RandomCrop(size=128),
    transforms.RandomRotation(degrees=10),
])

with torch.no_grad():
    for batch in train_data_loader:
        batch = batch[0]
        break

    batch = batch.to(device)
    batch = transform2(batch)
    original_batch = batch.clone()

    print('test')
    print('the same images')
    img1 = batch[0]
    features, decoded_img = autoencoder(img1.unsqueeze(0))

    img1 = img1.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('original', img1)

    img2 = decoded_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    cv2.imshow('reconstruction', img2)
    cv2.waitKey(0)


    original_batch.to(device)
    original_batch = transform2(original_batch)

    image = np.empty(shape=(batch_size, batch_size))
    for index1, index2 in itertools.product(range(batch_size), range(batch_size)):
        img1, img2 = original_batch[index1].unsqueeze(0), original_batch[index2].unsqueeze(0)

        features1, reconstruction1 = autoencoder(img1)
        features2, reconstruction2 = autoencoder(img2)

        result = siamese_network(features1, features2).item()
        result = 1 if result >= threshold else 0
        image[index1, index2] = result

    print(image.shape)
    print(image)
    # print(image.tolist())
    cv2.imshow('result', image)
    cv2.waitKey(0)