import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from train_data_functions import AllWeatherDataset
from utils import to_psnr, adjust_learning_rate
from perceptual import LossNetwork
from torchvision.models import vgg16

from transweather_masked import MaskedResidualTransWeather, MaskNet
# your real MaskNet (not dummy)

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-crop_size', default=[192, 192], nargs='+', type=int)
parser.add_argument('-train_batch_size', default=2, type=int)
parser.add_argument('-val_batch_size', default=1, type=int)
parser.add_argument('-num_epochs', default=10, type=int)
parser.add_argument('-lambda_loss', default=0.04, type=float)
parser.add_argument('-mask_weight', default=0.01, type=float)
parser.add_argument('-seed', default=19, type=int)
parser.add_argument('-vis_interval', default=500, type=int)

args = parser.parse_args()

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Directories
# -----------------------------
os.makedirs(args.exp_name, exist_ok=True)
os.makedirs(f"{args.exp_name}/mask_vis", exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
train_dataset = AllWeatherDataset(
    root='dataset/allweather',
    file_list='dataset/allweather/train.txt',
    crop_size=args.crop_size,
    train=True
)

val_dataset = AllWeatherDataset(
    root='dataset/allweather',
    file_list='dataset/allweather/val.txt',
    crop_size=args.crop_size,
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# Model
# -----------------------------
mask_net = MaskNet()
model = MaskedResidualTransWeather(mask_net)
model = model.to(device)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# -----------------------------
# Perceptual loss (VGG16)
# -----------------------------
vgg = vgg16(pretrained=True).features[:16].to(device)
for p in vgg.parameters():
    p.requires_grad = False
loss_network = LossNetwork(vgg)
loss_network.eval()

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(args.num_epochs):
    model.train()
    psnr_list = []
    start_time = time.time()

    adjust_learning_rate(optimizer, epoch)

    for i, (inp, gt, name) in enumerate(train_loader):
        inp = inp.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        # ---- Forward ----
        out, mask = model(inp)

        # ---- Losses ----
        recon_loss = F.smooth_l1_loss(out, gt)
        perceptual_loss = loss_network(out, gt)
        mask_loss = args.mask_weight * torch.mean(mask)

        loss = recon_loss + args.lambda_loss * perceptual_loss + mask_loss
        loss.backward()
        optimizer.step()

        # ---- Metrics ----
        psnr_list.extend(to_psnr(out, gt))

        # ---- Mask visualization ----
        if i % args.vis_interval == 0:
            save_image(mask[0], f"{args.exp_name}/mask_vis/e{epoch}_i{i}_mask.png")
            save_image(out[0],  f"{args.exp_name}/mask_vis/e{epoch}_i{i}_out.png")
            save_image(inp[0],  f"{args.exp_name}/mask_vis/e{epoch}_i{i}_inp.png")

            print(
                f"[Epoch {epoch} | Iter {i}] "
                f"Loss: {loss.item():.4f} | "
                f"Recon: {recon_loss.item():.4f} | "
                f"Mask(mean): {mask.mean().item():.4f}"
            )

    # ---- Epoch summary ----
    avg_psnr = sum(psnr_list) / len(psnr_list)
    epoch_time = time.time() - start_time

    print(
        f"\nEpoch [{epoch+1}/{args.num_epochs}] "
        f"PSNR: {avg_psnr:.2f} dB | "
        f"Time: {epoch_time:.1f}s\n"
    )

    # ---- Save checkpoint ----
    torch.save(model.state_dict(), f"{args.exp_name}/latest.pth")

print("✅ Training finished")
