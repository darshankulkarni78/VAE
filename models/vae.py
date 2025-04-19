import torch
import torch.nn.functional as F
import torch.nn as nn
from .encoder import encoder
from .decoder import decoder

class VAE(nn.Module):

  def __init__(self, latent_dim=512):
    super().__init__()
    self.enc = encoder(latent_dim=latent_dim)
    self.dec = decoder(latent_dim=latent_dim)

    # self.beta = nn.Parameter(torch.tensor(0.01))

  def forward(self, x):
    myu, logvar = self.enc(x)
    z = self.enc.reparam(myu, logvar)
    re_x = self.dec(z)

    return re_x, myu, logvar

  def loss(self, x, re_x, myu, logvar, **kwargs):
      # Reconstruction loss: per-sample average
      recon_loss = F.mse_loss(re_x, x, reduction='none')
      recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1)  # per-sample sum
      recon_loss = recon_loss.mean()  # average over batch

      # KL divergence: per-sample mean
      kl = 0.5 * torch.sum(torch.exp(logvar) + myu**2 - logvar - 1, dim=1)
      kl = kl.mean()

      # KL weight for annealing
      kl_weight = kwargs.get('kl_weight', 1)

      # Total VAE loss
      lvae = recon_loss + kl_weight * kl

      return kl.detach(), recon_loss.detach(), lvae