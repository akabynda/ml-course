import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from unet import ModifiedUNet, dice_loss, evaluate, train, jaccard_index
from dataset import ImageMaskDatasetWithAugmentation
import torch.optim as optim
import albumentations as A

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

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(p=0.5),
])

test_transform = A.Compose([
    A.Resize(256, 256),
])

train_dataset = ImageMaskDatasetWithAugmentation(train_image_files, train_mask_files, augmentation=train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = ImageMaskDatasetWithAugmentation(test_image_files, test_mask_files, augmentation=test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModifiedUNet(in_channels=3, exit_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = dice_loss

num_epochs = 20
best_test_iou = 0
for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch + 1}")
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_iou = evaluate(model, test_loader, device)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test IoU: {test_iou:.4f}')
    if test_iou > best_test_iou:
        best_test_iou = test_iou
        torch.save(model.state_dict(), "best_model_resnet18.pth")

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

            images_flipped = torch.flip(images, [3])
            outputs_flipped = model(images_flipped)
            outputs_flipped = torch.flip(outputs_flipped, [3])
            outputs_flipped = torch.sigmoid(outputs_flipped)
            outputs_flipped = (outputs_flipped > 0.5).float()
            iou_flipped = jaccard_index(outputs_flipped, masks).item()

            total_iou += (iou_normal + iou_flipped) / 2
    return total_iou / len(loader)

model.load_state_dict(torch.load("best_model_resnet18.pth"))
test_iou_tta = evaluate_tta(model, test_loader, device)
print(f'Test IoU with TTA: {test_iou_tta:.4f}')
