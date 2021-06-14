class Config:
    latent_size = 1024

    data_folder = './data/enhanced'
    saved_models_folder = './saved_models_autoencoder'

    # training
    epochs = 500
    batch_size = 100
    lr = 0.001
    save_frequency = 25
    load_model = True