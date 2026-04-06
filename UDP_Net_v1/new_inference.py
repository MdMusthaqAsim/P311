import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Import your model components
from transweather_masked import MaskedResidualTransWeather, MaskNet

def pad_image_to_32(img_tensor):
    """
    Pads the image tensor so that its height and width are multiples of 32.
    TransWeather's encoder downsamples by 4 * 2 * 2 * 2 = 32x.
    If not padded to 32, skip connections will violently misalign.
    """
    _, _, h, w = img_tensor.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    padded_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_tensor, h, w

def tensor_to_plot_img(tensor):
    tensor = tensor.squeeze(0).cpu().detach().permute(1, 2, 0)
    return tensor.numpy()

def main():
    # -----------------------------

    # Hardcoded Paths
    # -----------------------------
    input_path = r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\perhaps_25_epochs\mask_vis\e24_i650_inp.png"
    checkpoint_path = r"C:\Users\USER\IdeaProjects\P311\UDP_Net_v1\outputs\perhaps_25_epochs\latest.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load Model
    # -----------------------------
    print("Loading model architecture and weights...")
    mask_net = MaskNet()
    model = MaskedResidualTransWeather(mask_net)

    try:
        # strict=False is often required for custom wrapped models
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at '{checkpoint_path}'.")
        return

    model = model.to(device)

    # Set to eval to disable Dropouts
    model.eval()

    # CRITICAL FIX: The batch size of 2 during training corrupted the BatchNorm stats.
    # We must force BatchNorm to calculate stats dynamically during inference.
    for m in model.mask_net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    # -----------------------------
    # Load and Preprocess Image
    # -----------------------------
    print(f"Processing image: {input_path}")
    try:
        img = Image.open(input_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Input image not found at '{input_path}'.")
        return

    # Convert to tensor [C, H, W] in [0.0, 1.0] range
    inp_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    # Pad to multiple of 32 for the Transformer skip connections
    inp_padded, orig_h, orig_w = pad_image_to_32(inp_tensor)

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.no_grad():
        out_padded, mask_padded = model(inp_padded)

        # Crop back to original dimensions
        out = out_padded[:, :, :orig_h, :orig_w]
        mask = mask_padded[:, :, :orig_h, :orig_w]
        inp_vis = inp_tensor # The original unpadded input

        # Clamp to valid image range
        out = torch.clamp(out, 0.0, 1.0)
        mask = torch.clamp(mask, 0.0, 1.0)

    # -----------------------------
    # Plotting
    # -----------------------------
    print(f"Mask Min/Max values: {mask.min().item():.4f} / {mask.max().item():.4f}")
    if mask.max().item() < 0.05:
        print("WARNING: The mask is still near zero! The model is choosing not to apply dehazing.")

    print("Generating plot...")

    np_inp = tensor_to_plot_img(inp_vis)
    np_out = tensor_to_plot_img(out)
    np_mask = tensor_to_plot_img(mask).squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np_inp)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Using vmin=0, vmax=1 ensures the grayscale plot shows true values,
    # not artificially contrasted values.
    axes[1].imshow(np_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    axes[2].imshow(np_out)
    axes[2].set_title("Restored Output")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()