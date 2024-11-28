#!/usr/bin/env python
# coding: utf-8

# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashishjangra27/doodle-dataset")

print("Path to dataset files:", path)


# In[2]:


get_ipython().system('pip install Kagglehub')


# In[ ]:


print(5)


# In[7]:


get_ipython().system(' pip install torch')


# In[10]:


get_ipython().system('pip install torchvision')


# In[12]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# Device configuration
device = torch.device("cpu")


# In[21]:


import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Path to your dataset directory
data_dir = "/root/.cache/kagglehub/datasets/ashishjangra27/doodle-dataset/versions/1/doodle"

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize(64),           # Resize images to 64x64
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale (1 channel)
    transforms.ToTensor(),           # Convert to PyTorch tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])


# Load the dataset
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


# In[22]:


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        return self.model(x)


# In[23]:


print(dataset)


# In[24]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


# In[25]:


latent_dim = 100

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# In[ ]:


num_epochs = 1
sample_dir = "./samples"

# Create sample directory
os.makedirs(sample_dir, exist_ok=True)

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        outputs = discriminator(real_images).view(-1, 1)
        d_loss_real = criterion(outputs, real_labels)

        # Generate fake images
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach()).view(-1, 1)
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images).view(-1, 1)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Logging
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save generated samples
    if (epoch + 1) % 10 == 0:
        save_image(fake_images.data[:25], os.path.join(sample_dir, f"epoch_{epoch+1}.png"), nrow=5, normalize=True)


# In[ ]:


torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")


# In[ ]:




