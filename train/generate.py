import torch
import matplotlib.pyplot as plt
import numpy as np
from models.vae import VAE 

def load_model_from_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return model, optimizer

def generate_samples(checkpoint_path, device, latent_dim=512, num_samples=8, save_path=None):
    model = VAE(latent_dim=latent_dim).to(device)
    
    model, _ = load_model_from_checkpoint(checkpoint_path, model, optimizer=None, device=device)

    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.dec(z).cpu()

        plt.figure(figsize=(10, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(np.clip(samples[i].permute(1, 2, 0).numpy(), 0, 1))
            plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()