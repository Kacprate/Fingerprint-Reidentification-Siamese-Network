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

batch_size = 100
threshold = 0.7

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

autoencoder = AutoEncoder(config)
siamese_network = SiameseNetwork(config)

autoencoder_file = '/autoencoder_epoch225_loss0.7295.pth'
siamese_file = '/siamese_network_epoch225_loss0.7295.pth'

autoencoder.load_state_dict(torch.load(config.saved_models_folder + autoencoder_file))
autoencoder.to(device)
autoencoder.train()

siamese_network.load_state_dict(torch.load(config.saved_models_folder + siamese_file))
siamese_network.to(device)
siamese_network.train()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(config.data_folder, transform=transform)
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

transform2 = transforms.Compose([
    transforms.RandomCrop(size=128),
    transforms.RandomRotation(degrees=10),
])

with torch.no_grad():
    for batch in train_data_loader:
        batch = batch[0]
        break

    original_batch = batch.clone()

    original_batch = original_batch.to(device)

    # batch = batch.to(device)

    # batch = transform2(batch)


    batch_var1 = transform2(original_batch)
    batch_var2 = transform2(original_batch)

    features1, reconstructed_images1 = autoencoder(batch_var1)
    features2, _ = autoencoder(batch_var2)
    bad_features = features1.roll(shifts=1, dims=0)

    result_ok = siamese_network(features1, features2)
    result_not_ok = siamese_network(features1, bad_features)

    counter = 0
    for r in result_ok.cpu().numpy():
        if r > threshold:
            counter += 1

    for r in result_not_ok.cpu().numpy():
        if r < threshold:
            counter += 1

    print(result_ok)
    print(result_not_ok)
    print(f'correctly matched {counter} fingerprints out of {batch_size * 2} samples')

    img1 = batch_var1[0]
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('original', img1)

    img2 = reconstructed_images1[0].permute(1, 2, 0).cpu().numpy()
    cv2.imshow('reconstruction', img2)
    cv2.waitKey(0)





    # batch = transform2(batch)
    # batch = batch.to(device)

    # features1, _ = autoencoder(batch)

    # results = []
    # for index in range(batch_size):
    #     image_xd = batch[index]
    #     batch2 = image_xd.unsqueeze(0).repeat((batch_size, 1, 1, 1))
    #     batch2.to(device)

    #     features2, _ = autoencoder(batch2)
    #     result = siamese_network(features1, features2)

    #     # if index == 0:
    #     #     print((result.cpu().numpy() * 100).astype(int))
    #     #     raise Exception

    #     result[result >= threshold] = 1
    #     result[result < threshold] = 0

    #     results.append(result.cpu().detach().numpy())
    #     # print(result)

    # image = np.array(results)
    # print(image.shape)
    # # print(image.tolist())
    # cv2.imshow('result', image)
    # cv2.waitKey(0)









    # original_batch = transform2(original_batch)

    # mean = original_batch.mean()
    # std = original_batch.std()
    # print(mean, std)
    # original_batch = (original_batch - mean) / std

    # image = np.empty(shape=(batch_size, batch_size))

    # for index1 in range(batch_size):
    #     for index2 in range(batch_size):
    #         img1 = original_batch[index1].repeat(2, 1, 1, 1)
    #         img2 = original_batch[index2].repeat(2, 1, 1, 1)

    #         features1, reconstruction1 = autoencoder(img1)
    #         features2, reconstruction2 = autoencoder(img2)

    #         result = siamese_network(features1, features2)[0].item()

    #         if index1 == index2:
    #             print(result)

    #         result = 1 if result >= threshold else 0
    #         image[index1, index2] = result

    # print(image.shape)
    # print(image)
    # # print(image.tolist())
    # cv2.imshow('result', image)
    # cv2.waitKey(0)