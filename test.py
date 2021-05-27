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
from models import Encoder, SiameseNetwork

batch_size = 8
threshold = 0.9

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

encoder = Encoder(config)
siamese_network = SiameseNetwork(config)

encoder.load_state_dict(torch.load(config.saved_models_folder + '/encoder_epoch500_loss0.0009.pth'))
encoder.to(device)
encoder.eval()

siamese_network.load_state_dict(torch.load(config.saved_models_folder + '/siamese_network_epoch500_loss0.0009.pth'))
siamese_network.to(device)
siamese_network.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(config.data_folder, transform=transform)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

transform2 = transforms.Compose([
    # transforms.RandomCrop(size=128),
    transforms.RandomRotation(degrees=10),
])

with torch.no_grad():
    for batch in train_data_loader:
        batch = batch[0]
        break

    batch = batch.to(device)
    original_batch = batch.clone()

    # batch = transform2(batch)

    # features1 = encoder(batch)

    # results = []
    # for index in range(batch_size):
    #     batch2 = original_batch[index]
    #     batch2 = batch2.unsqueeze(0).repeat((batch_size, 1, 1, 1))
    #     batch2.to(device)

    #     batch2 = transform2(batch2)

    #     features2 = encoder(batch2)
    #     result = siamese_network(features1, features2)

    #     # if index == 0:
    #     #     print((result.cpu().numpy() * 100).astype(int))
    #     #     raise Exception

    #     # result[result >= threshold] = 1
    #     # result[result < threshold] = 0

    #     results.append(result.cpu().numpy())
    #     # print(result)

    # image = np.array(results)
    # print(image.shape)
    # # print(image.tolist())
    # cv2.imshow('result', image)
    # cv2.waitKey(0)

    original_batch.to(device)
    original_batch = transform2(original_batch)

    image = np.empty(shape=(batch_size, batch_size))
    for index1, index2 in itertools.product(range(batch_size), range(batch_size)):
        img1, img2 = original_batch[index1].unsqueeze(0), original_batch[index2].unsqueeze(0)

        features1, features2 = encoder(img1), encoder(img2)
        result = siamese_network(features1, features2)
        image[index1, index2] = result.item()

    print(image.shape)
    # print(image.tolist())
    cv2.imshow('result', image)
    cv2.waitKey(0)