# Variational Autoencoder (VAE) Project

This repository contains the implementation of a Variational Autoencoder (VAE) for image generation. The project is structured into multiple modules for data loading, model definition, training, and sample generation.

## Project Structure

### File Descriptions

- **`main.py`**: The main script to load data, train the VAE model, and generate samples.
- **`data/data_loader.py`**: Contains the `load_data` function to preprocess and load the dataset.
- **`models/encoder.py`**: Defines the encoder part of the VAE.
- **`models/decoder.py`**: Defines the decoder part of the VAE.
- **`models/vae.py`**: Combines the encoder and decoder into the VAE model and defines the loss function.
- **`train/trainer.py`**: Contains the `train_vae` function for training the VAE and `save_model` for saving the trained model.
- **`train/generate.py`**: Contains functions to load a trained model and generate image samples.

## Dataset

The dataset used for training is 50k CelebA Dataset (64x64). You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/therealcyberlord/50k-celeba-dataset-64x64). After downloading, place the dataset in a folder and update the path in `main.py`.
