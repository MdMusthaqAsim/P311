import torch
import numpy as np
from torch.utils.data import DataLoader
from train_data_functions import AllWeatherDataset

from transweather_model import Transweather          # original model file
from transweather_masked import MaskedResidualTransWeather, MaskNet

# ----------------------------------------------------
# masked PSNR
# ----------------------------------------------------
def masked_psnr(pred, gt, mask, eps=1e-8):
    # pred, gt : [B,3,H,W] in [0,1]
    # mask     : [B,1,H,W]   (soft mask allowed)

    diff2 = (pred - gt) ** 2

    # broadcast mask to RGB
    mask = mask.expand_as(diff2)

    mse = (diff2 * mask).sum() / (mask.sum() * 1.0 + eps)

    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr


# ----------------------------------------------------
# main
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- models ----------------

# MaskNet (trained)
mask_net = MaskNet().to(device)

# masked model
masked_model = MaskedResidualTransWeather(mask_net).to(device)

# baseline TransWeather
base_model = Transweather().to(device)

# ---------------- load checkpoints ----------------

masked_ckpt = torch.load("masked_baseline/latest.pth", map_location=device)
masked_model.load_state_dict(masked_ckpt, strict=False)

base_ckpt = torch.load("pretrain_tw/best", map_location=device)
base_model.load_state_dict(base_ckpt, strict=False)
#base_model.load(base_ckpt) if hasattr(base_model, "load") else base_model.load_state_dict(base_ckpt, strict=False)

masked_model.eval()
base_model.eval()
mask_net.eval()

# ---------------- data ----------------

val_dataset = AllWeatherDataset(
    root="dataset/allweather",
    file_list="dataset/allweather/val.txt",
    crop_size=[192,192],
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ---------------- loop ----------------

masked_psnr_base = []
masked_psnr_masked = []
mask_means = []

with torch.no_grad():

    for i, batch in enumerate(val_loader):

        inp, gt, _ = batch
        inp = inp.to(device)
        gt  = gt.to(device)

        # ----- masked model -----
        pred_masked, mask = masked_model(inp)

        # make sure mask is pixel mask
        # (many of your MaskNet versions output 1xHxW already)
        if mask.ndim == 2:
            raise RuntimeError("Mask must be spatial (B,1,H,W), not patch vector")

        # ----- baseline model -----
        pred_base = base_model(inp)

        # ----- masked PSNRs -----
        psnr_m = masked_psnr(pred_masked, gt, mask)
        psnr_b = masked_psnr(pred_base,   gt, mask)

        masked_psnr_masked.append(psnr_m.item())
        masked_psnr_base.append(psnr_b.item())
        mask_means.append(mask.mean().item())

        if i % 50 == 0:
            print(f"{i}/{len(val_loader)}")

# ---------------- results ----------------

print("\n=========== MASKED REGION COMPARISON ===========")
print(f"Masked PSNR (TransWeather)        : {np.mean(masked_psnr_base):.2f} dB")
print(f"Masked PSNR (Masked TransWeather) : {np.mean(masked_psnr_masked):.2f} dB")
print(f"Mean mask value                   : {np.mean(mask_means):.4f}")
print("===============================================\n")