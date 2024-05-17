import torch
import torch.nn as nn
from torchvision import models


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq_block(x)


class CNNBlocks(nn.Module):
    def __init__(self, n_conv, in_channels, out_channels, padding):
        super(CNNBlocks, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_conv):
            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PretrainedEncoder(nn.Module):
    def __init__(self, in_channels):
        super(PretrainedEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool),
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ])

    def forward(self, x):
        connections = []
        for layer in self.encoder_layers:
            x = layer(x)
            # print(f'Encoder Layer Output Shape: {x.shape}')
            connections.append(x)
        return x, connections


class ModifiedDecoder(nn.Module):
    def __init__(self, in_channels, exit_channels):
        super(ModifiedDecoder, self).__init__()
        self.layers = nn.ModuleList()

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])

        self.dec_blocks = nn.ModuleList([
            CNNBlocks(2, 1024, 512, padding=1),
            CNNBlocks(2, 512, 256, padding=1),
            CNNBlocks(2, 256, 128, padding=1),
            CNNBlocks(2, 128, 64, padding=1),
        ])

        self.final_conv = nn.Conv2d(64, exit_channels, kernel_size=1)
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x, connections):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            conn = connections.pop()
            if x.size() != conn.size():
                x = nn.functional.interpolate(x, size=conn.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, conn], dim=1)
            x = self.dec_blocks[i](x)
        x = self.final_conv(x)
        x = self.final_upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, n_down=4):
        super(Encoder, self).__init__()

        self.enc_layers = nn.ModuleList()
        for _ in range(n_down):
            self.enc_layers += [
                CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
                nn.MaxPool2d(2, 2)]

            in_channels = out_channels
            out_channels = 2 * out_channels

        self.enc_layers.append(CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding))

    def forward(self, x):
        connections = []
        for layer in self.enc_layers:
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                connections.append(x)
            else:
                x = layer(x)
        return x, connections


class ModifiedUNet(nn.Module):
    def __init__(self, in_channels, exit_channels):
        super(ModifiedUNet, self).__init__()
        self.encoder = PretrainedEncoder(in_channels)
        self.decoder = ModifiedDecoder(512, exit_channels)  # 512 is the output channels of the ResNet18 encoder

    def forward(self, x):
        enc_out, connections = self.encoder(x)
        dec_out = self.decoder(enc_out, connections)
        return dec_out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, exit_channels, padding, n_up=4):
        super(Decoder, self).__init__()

        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()

        for i in range(n_up):
            self.layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                            CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding)]

            in_channels //= 2
            out_channels //= 2

        self.layers.append(nn.Conv2d(in_channels, self.exit_channels, kernel_size=1, padding=0))

    def forward(self, x, connections):
        connections.pop(-1)
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):
                connections_current = connections.pop(-1)
                x = torch.cat([x, connections_current], dim=1)
                x = layer(x)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, first_out_channels, exit_channels, n_down=4, padding=1):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, first_out_channels, padding=padding, n_down=n_down)
        self.decoder = Decoder(first_out_channels * (2 ** n_down),
                               first_out_channels * (2 ** (n_down - 1)),
                               exit_channels, padding=padding, n_up=n_down)

    def forward(self, x):
        enc_out, connections = self.encoder(x)
        dec_out = self.decoder(enc_out, connections)
        return dec_out


def jaccard_index(pred, target):
    pred = pred.to(torch.int)  # Convert to binary integers
    target = target.to(torch.int)  # Convert to binary integers
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            total_iou += jaccard_index(outputs, masks).item()
    return total_iou / len(loader)
