# main_train.py with Teacher-Student Knowledge Distillation and Checkpoint Support

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from utils.dataset import BlurSharpDataset
from models.student_model import StudentCNN
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Dataset
dataset = BlurSharpDataset("data/train/input", "data/train/target")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Student Model
student = StudentCNN().to(device)

# Teacher Model (Pretrained ResNet18 used as image feature extractor)
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
teacher = resnet18(weights=weights)
teacher.fc = nn.Identity()
teacher = teacher.to(device)
teacher.eval()

# Loss functions
mse_loss = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

# Resize to feed into ResNet
resize = transforms.Resize((224, 224))

# Training
num_epochs = 200
for epoch in range(num_epochs):
    total_loss = 0
    for blurred, sharp in dataloader:
        blurred = blurred.to(device)
        sharp = sharp.to(device)

        output = student(blurred)

        pixel_loss = mse_loss(output, sharp)

        with torch.no_grad():
            t_feats = teacher(resize(sharp))
        s_feats = teacher(resize(output))
        distill_loss = mse_loss(s_feats, t_feats)

        loss = pixel_loss + 0.1 * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # SSIM evaluation
    student.eval()
    with torch.no_grad():
        sample_blur, sample_target = next(iter(dataloader))
        sample_blur = sample_blur.to(device)
        sample_target = sample_target.to(device)
        sample_output = student(sample_blur)

        output_img = sample_output[0].cpu().numpy().transpose(1, 2, 0)
        target_img = sample_target[0].cpu().numpy().transpose(1, 2, 0)
        ssim_val = ssim(target_img, output_img, channel_axis=2, data_range=1.0)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, SSIM: {ssim_val:.4f}")
    student.train()

    if (epoch + 1) % 10 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(student.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

torch.save(student.state_dict(), "student_model_384x384_e200.pth")
print("✅ Training complete, distilled model saved!")
