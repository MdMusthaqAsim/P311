import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from transweather_masked import MaskedResidualTransWeather, MaskNet

# ---------------- config ----------------
N_STEPS = 10
# change as needed

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- model ----------------
mask_net = MaskNet()
model = MaskedResidualTransWeather(mask_net).to(device)

ckpt = torch.load(r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\one_last_10_epoch\latest.pth", map_location=device)
model.load_state_dict(ckpt, strict=False)
model.eval()

# ---------------- image ----------------
input_path = r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\testImg3.jpeg"

transform = T.Compose([
    T.Resize((192,192)),
    T.ToTensor()
])

inp = transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)

# ---------------- helper ----------------
def to_img(x):
    x = x.squeeze(0).detach().cpu().permute(1,2,0).numpy()
    return np.clip(x, 0, 1)

# ---------------- loop ----------------
current = inp.clone()

inputs  = []
outputs = []
masks   = []
overlays= []

for step in range(N_STEPS):

    with torch.no_grad():
        out, mask = model(current)

    # metrics
    mask_mean = mask.mean().item()
    mask_area = (mask > 0.5).float().mean().item()

    print(f"Step {step+1} | mask_mean={mask_mean:.4f} | area={mask_area:.4f}")

    # convert
    inp_img = to_img(current)
    out_img = to_img(out)
    mask_img = mask.squeeze().detach().cpu().numpy()

    overlay = inp_img * (1 - mask_img[..., None]) + mask_img[..., None]

    # store
    inputs.append(inp_img)
    outputs.append(out_img)
    masks.append(mask_img)
    overlays.append(overlay)

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
    plt.imshow(masks[i], cmap="gray")
    plt.title(f"Mask {i+1}")
    plt.axis("off")

    # Overlay
    plt.subplot(rows, cols, 3*cols + i + 1)
    plt.imshow(overlays[i])
    plt.title(f"Overlay {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()