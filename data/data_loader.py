import os
import cv2
from torchvision import transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    image = []
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        image.append(img_tensor)

    train_imgs, test_imgs = train_test_split(image, test_size=0.2, random_state=36)
    train_dataset = TensorDataset(torch.stack(train_imgs))
    test_dataset = TensorDataset(torch.stack(test_imgs))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader