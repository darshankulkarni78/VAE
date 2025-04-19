import torch
from models.vae import VAE
from data.data_loader import load_data
from train.trainer import train_vae
from train.generate import generate_samples
# from train.trainer import save_model    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = load_data("/content/50k", batch_size=32)
vae_model = VAE(latent_dim=512).to(device)
vae_model, optimiser = train_vae(vae_model, train_loader, test_loader, epochs=20, device=device)
generate_samples(vae_model, device=device, latent_dim=512, num_samples=8)

# save_model(vae_model, optimiser, "vae_model.pth")