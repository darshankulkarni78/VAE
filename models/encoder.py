import torch.nn as nn
import torch

class encoder(nn.Module):

  def __init__(self, latent_dim=512):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
    self.bn4 = nn.BatchNorm2d(256)
    self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
    self.bn5 = nn.BatchNorm2d(512)
    self.Lrelu = nn.LeakyReLU()

    self.flt1 = nn.Flatten(start_dim=1)

    self.fc1 = nn.Linear(512*2*2, 1024)
    # self.fc2 = nn.Linear(2048, 512)

    self.myu = nn.Linear(1024, latent_dim)
    self.logvar = nn.Linear(1024, latent_dim)

  def forward(self, x):
    x = self.Lrelu(self.bn1(self.conv1(x)))
    x = self.Lrelu(self.bn2(self.conv2(x)))
    x = self.Lrelu(self.bn3(self.conv3(x)))
    x = self.Lrelu(self.bn4(self.conv4(x)))
    x = self.Lrelu(self.bn5(self.conv5(x)))

    x = self.flt1(x)
    x = self.fc1(x)
    # x = self.fc2(x)

    myu = self.myu(x)
    logvar = self.logvar(x)

    # z = self.reparam(myu, logvar)

    return myu, logvar

  def reparam(self, myu, logvar):
    epsilon = torch.randn_like(myu)
    std = torch.exp(0.5*logvar)

    z = myu + std*epsilon

    return z