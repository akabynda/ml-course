import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import ImageMaskDataset
from unet import UNet, dice_loss, evaluate, train
import torch
import torch.optim as optim

image_dir = 'dataset/images'
mask_dir = 'dataset/masks'

image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

indices = range(len(image_files))
train_indices, test_indices = train_test_split(list(indices), test_size=0.2, random_state=42)

train_image_files = [image_files[i] for i in train_indices]
train_mask_files = [mask_files[i] for i in train_indices]
test_image_files = [image_files[i] for i in test_indices]
test_mask_files = [mask_files[i] for i in test_indices]

train_dataset = ImageMaskDataset(list(train_image_files), list(train_mask_files))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = ImageMaskDataset(list(test_image_files), list(test_mask_files))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, first_out_channels=64, exit_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = dice_loss

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch + 1}")
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_iou = evaluate(model, test_loader, device)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test IoU: {test_iou:.4f}')
