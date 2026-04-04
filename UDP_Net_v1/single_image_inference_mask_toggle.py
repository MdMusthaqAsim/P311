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

ckpt = torch.load(
    r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\perhaps_25_epochs\latest.pth",
    map_location=device
)
model.load_state_dict(ckpt, strict=False)
model.eval()

# ---------------- image ----------------
input_path = r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\testImg3.jpeg"

transform = T.Compose([
    T.Resize((192, 192)),
    T.ToTensor()
])

inp = transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)

# ---------------- create custom masks ----------------
B, C, H, W = inp.shape

mask_all_ones  = torch.ones((B, 1, H, W), device=device)
mask_all_zeros = torch.zeros((B, 1, H, W), device=device)
mask_half      = torch.full((B, 1, H, W), 0.5, device=device)

# ---------------- inference ----------------
with torch.no_grad():
    # normal mask (MaskNet)
    out_normal, _ = model(inp)

    # fully ON mask
    out_on, _ = model(inp, custom_mask=mask_all_ones)

    # half mask
    out_half, _ = model(inp, custom_mask=mask_half)

    # fully OFF mask
    out_off, _ = model(inp, custom_mask=mask_all_zeros)

# ---------------- helper ----------------
def to_img(x):
    x = x.squeeze(0).cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0, 1)

# ---------------- display ----------------
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Normal")
plt.imshow(to_img(out_normal))
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Mask = 1")
plt.imshow(to_img(out_on))
plt.axis("off")

plt.subplot(1,4,3)
plt.title("Mask = 0.5")
plt.imshow(to_img(out_half))
plt.axis("off")

plt.subplot(1,4,4)
plt.title("Mask = 0")
plt.imshow(to_img(out_off))
plt.axis("off")

plt.tight_layout()
plt.show()