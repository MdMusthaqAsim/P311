import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from transweather_masked import MaskedResidualTransWeather, MaskNet

# ---------------- config ----------------
N_STEPS = 5
# change as needed

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- helper ----------------
def pad_image_to_32(img_tensor):
    """
    Pads the image tensor so that its height and width are multiples of 32.
    TransWeather's encoder downsamples by 4 * 2 * 2 * 2 = 32x.
    """
    _, _, h, w = img_tensor.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    padded_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_tensor, h, w

def to_img(x):
    x = x.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0, 1)

# ---------------- model ----------------
print("Loading model architecture and weights...")
mask_net = MaskNet()
model = MaskedResidualTransWeather(mask_net).to(device)

# Updated to use the checkpoint path from your working script
ckpt = torch.load(r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\one_last_10_epoch\latest.pth", map_location=device)
model.load_state_dict(ckpt, strict=False)

# Set to eval to disable Dropouts
model.eval()

# CRITICAL FIX: Force BatchNorm to calculate stats dynamically during inference
for m in model.mask_net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.train()

# ---------------- image ----------------
input_path = r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\perhaps_25_epochs\mask_vis\e24_i650_inp.png"
print(f"Processing image: {input_path}")

# Removed the 192x192 Resize to maintain original resolution
transform = T.Compose([
    T.ToTensor()
])

inp = transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)

# ---------------- loop ----------------
current = inp.clone()

inputs  = []
outputs = []
masks   = []
overlays= []

for step in range(N_STEPS):

    # 1. Pad the current image before passing to the model
    current_padded, orig_h, orig_w = pad_image_to_32(current)

    with torch.no_grad():
        out_padded, mask_padded = model(current_padded)

    # 2. Crop the output and mask back to original dimensions
    out = out_padded[:, :, :orig_h, :orig_w]
    mask = mask_padded[:, :, :orig_h, :orig_w]

    # 3. Clamp to valid ranges
    out = torch.clamp(out, 0.0, 1.0)
    mask = torch.clamp(mask, 0.0, 1.0)

    # metrics
    mask_mean = mask.mean().item()
    mask_area = (mask > 0.5).float().mean().item()

    print(f"Step {step+1} | mask_mean={mask_mean:.4f} | area={mask_area:.4f}")

    # convert
    inp_img = to_img(current)
    out_img = to_img(out)
    mask_img = mask.squeeze(0).squeeze(0).detach().cpu().numpy() # [H, W]

    # create a visual overlay (adds a white tint where mask is active)
    overlay = inp_img * (1 - mask_img[..., None]) + mask_img[..., None]
    overlay = np.clip(overlay, 0, 1)

    # store
    inputs.append(inp_img)
    outputs.append(out_img)
    masks.append(mask_img)
    overlays.append(overlay)

    # Update current for the next iteration
    current = out.clone()

# ---------------- plotting ----------------
rows = 4   # Input / Output / Mask / Overlay
cols = N_STEPS

plt.figure(figsize=(4*cols, 10))

for i in range(N_STEPS):

    # Input
    plt.subplot(rows, cols, i + 1)
    plt.imshow(inputs[i])
    plt.title(f"Input {i+1}")
    plt.axis("off")

    # Output
    plt.subplot(rows, cols, cols + i + 1)
    plt.imshow(outputs[i])
    plt.title(f"Output {i+1}")
    plt.axis("off")

    # Mask
    plt.subplot(rows, cols, 2*cols + i + 1)
    plt.imshow(masks[i], cmap="gray", vmin=0, vmax=1)
    plt.title(f"Mask {i+1}")
    plt.axis("off")

    # Overlay
    plt.subplot(rows, cols, 3*cols + i + 1)
    plt.imshow(overlays[i])
    plt.title(f"Overlay {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()