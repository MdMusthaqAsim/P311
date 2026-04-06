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
input_path = r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\tt4.jpeg"

transform = T.Compose([
    T.Resize((192, 192)),
    T.ToTensor()
])

inp = transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)

# ---------------- inference ----------------
with torch.no_grad():
    out, _ = model(inp)

# ---------------- tensor -> image ----------------
out_img = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
out_img = np.clip(out_img, 0, 1)

# ---------------- display ----------------
plt.imshow(out_img)
plt.axis("off")
plt.title("Output")
plt.show()