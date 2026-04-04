import torch
import numpy as np
from torch.utils.data import DataLoader

from train_data_functions import AllWeatherDataset
from utils import to_psnr

from transweather_masked import MaskedResidualTransWeather, MaskNet

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- model ----------------
mask_net = MaskNet()
model = MaskedResidualTransWeather(mask_net).to(device)

ckpt = torch.load(r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\perhaps_25_epochs\latest.pth", map_location=device)
model.load_state_dict(ckpt, strict=False)
model.eval()

# ---------------- dataset ----------------
val_dataset = AllWeatherDataset(
    root='dataset/allweather',
    file_list='dataset/allweather/val.txt',
    crop_size=[192,192],
    train=False
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ---------------- loop ----------------
psnr_list = []
mask_means = []

with torch.no_grad():
    for i, (inp, gt, _) in enumerate(val_loader):

        inp = inp.to(device)
        gt  = gt.to(device)

        out, mask = model(inp)

        psnr_list.extend(to_psnr(out, gt))
        mask_means.append(mask.mean().item())

        if i % 50 == 0:
            print(f"{i}/{len(val_loader)}")

# ---------------- results ----------------
print("\n=========== EVAL ===========")
print(f"PSNR        : {np.mean(psnr_list):.2f} dB")
print(f"Mask (mean) : {np.mean(mask_means):.4f}")
print("================================\n")