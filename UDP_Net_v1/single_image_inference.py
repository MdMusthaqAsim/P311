import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from transweather_masked import MaskedResidualTransWeather, MaskNet

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- model ----------------
mask_net = MaskNet()
model = MaskedResidualTransWeather(mask_net).to(device)

ckpt = torch.load(r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\perhaps_25_epochs\latest.pth", map_location=device)
model.load_state_dict(ckpt, strict=False)

model.eval()

# ---------------- image ----------------
input_path = r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\testImg3.jpeg"

transform = T.Compose([
    T.Resize((192,192)),
    T.ToTensor()
])

inp = transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)

# ---------------- inference ----------------
with torch.no_grad():
    out, mask = model(inp)

# ---------------- metrics ----------------
mask_mean = mask.mean().item()
mask_min  = mask.min().item()
mask_max  = mask.max().item()

# useful proxy
mask_area = (mask > 0.5).float().mean().item()

print("\n=========== INFERENCE ===========")
print(f"Mask mean : {mask_mean:.4f}")
print(f"Mask min  : {mask_min:.4f}")
print(f"Mask max  : {mask_max:.4f}")
print(f"Mask area (>0.5): {mask_area:.4f}")
print("================================\n")

# ---------------- tensor -> image ----------------
def to_img(x):
    x = x.squeeze(0).detach().cpu().permute(1,2,0).numpy()
    return np.clip(x, 0, 1)

inp_img = to_img(inp)
out_img = to_img(out)

mask_gray = mask.squeeze().detach().cpu().numpy()

# overlay (🔥 best debug tool)
overlay = inp_img * (1 - mask_gray[..., None]) + mask_gray[..., None]

# ---------------- plotting ----------------
plt.figure(figsize=(12,6))

# input
plt.subplot(2,3,1)
plt.title("Input")
plt.imshow(inp_img)
plt.axis("off")

# output
plt.subplot(2,3,2)
plt.title("Output")
plt.imshow(out_img)
plt.axis("off")

# mask
plt.subplot(2,3,3)
plt.title("Mask")
plt.imshow(mask_gray, cmap="gray")
plt.axis("off")

# overlay
plt.subplot(2,3,4)
plt.title("Mask Overlay")
plt.imshow(overlay)
plt.axis("off")

# metrics plot
plt.subplot(2,3,5)
metrics = [mask_mean, mask_min, mask_max, mask_area]
labels  = ["mean", "min", "max", "area>0.5"]

plt.bar(labels, metrics)
plt.title("Mask Metrics")

plt.tight_layout()
plt.show()