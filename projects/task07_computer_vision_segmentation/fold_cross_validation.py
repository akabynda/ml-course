import os
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageMaskDatasetWithAugmentation
from unet import UNet, dice_loss, evaluate, train, jaccard_index

image_dir = 'dataset/images'
mask_dir = 'dataset/masks'

# File retrieval
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

# Splitting data
indices = range(len(image_files))
train_indices, test_indices = train_test_split(list(indices), test_size=0.2, random_state=42)

train_image_files = [image_files[i] for i in train_indices]
train_mask_files = [mask_files[i] for i in train_indices]
test_image_files = [image_files[i] for i in test_indices]
test_mask_files = [mask_files[i] for i in test_indices]

# Augmentation and Transformations
train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(p=0.5)
])

test_transform = A.Compose([
    A.Resize(256, 256)
])

# Create test dataset and loader
test_dataset = ImageMaskDatasetWithAugmentation(test_image_files, test_mask_files, augmentation=test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(kf.split(train_image_files))

# Arrays to store metrics
train_losses = []
val_iou_scores = []

# Training and validation loop for each fold
for fold, (train_idx, val_idx) in enumerate(fold_indices):
    print(f"Starting fold {fold + 1}")

    # Create train and validation sets for this fold
    fold_train_image_files = [train_image_files[i] for i in train_idx]
    fold_train_mask_files = [train_mask_files[i] for i in train_idx]
    fold_val_image_files = [train_image_files[i] for i in val_idx]
    fold_val_mask_files = [train_mask_files[i] for i in val_idx]

    # Create datasets and loaders
    train_dataset = ImageMaskDatasetWithAugmentation(fold_train_image_files, fold_train_mask_files, augmentation=train_transform)
    val_dataset = ImageMaskDatasetWithAugmentation(fold_val_image_files, fold_val_mask_files, augmentation=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, first_out_channels=64, exit_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train and validate for the given number of epochs
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}")
        train_loss = train(model, train_loader, optimizer, dice_loss, device)
        val_iou = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_iou_scores.append(val_iou)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}')

# Plot learning curves
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r', label='Train Loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, val_iou_scores, 'b', label='Validation IoU')
plt.title('Validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

# Save and evaluate the aggregated model
torch.save(model.state_dict(), "model_fold.pth")

def evaluate_tta(model, loader, device):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            iou_normal = jaccard_index(outputs, masks).item()

            # Horizontal flip
            images_flipped = torch.flip(images, [3])
            outputs_flipped = model(images_flipped)
            outputs_flipped = torch.flip(outputs_flipped, [3])
            outputs_flipped = torch.sigmoid(outputs_flipped)
            outputs_flipped = (outputs_flipped > 0.5).float()
            iou_flipped = jaccard_index(outputs_flipped, masks).item()

            # Average the IoUs
            total_iou += (iou_normal + iou_flipped) / 2
    return total_iou / len(loader)

model.load_state_dict(torch.load("model_fold.pth"))
test_iou_tta = evaluate_tta(model, test_loader, device)
print(f'Test IoU with TTA: {test_iou_tta:.4f}')
