import torch.nn as nn
import torch

class decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)

        self.convt1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.convt2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.convt3 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.convt4 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.convt5 = nn.ConvTranspose2d(32, 3, 3, 2, 1, 1)

        self.Lrelu = nn.LeakyReLU()
        # self.final = nn.Conv2d(32, 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = z.view(-1, 512, 2, 2)

        z = self.Lrelu(self.convt1(z))
        z = self.Lrelu(self.convt2(z))
        z = self.Lrelu(self.convt3(z))
        z = self.Lrelu(self.convt4(z))
        # z = self.Lrelu(self.convt5(z))
        z = self.sigmoid(self.convt5(z))

        # z = self.sigmoid(z)

        return z