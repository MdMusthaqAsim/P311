import torch
import math
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def to_psnr(pred, gt):
    pred = pred.clamp(0, 1).detach().cpu().numpy()
    gt = gt.clamp(0, 1).detach().cpu().numpy()

    psnr_list = []
    for i in range(pred.shape[0]):
        psnr_list.append(psnr(gt[i], pred[i], data_range=1.0))
    return psnr_list


def print_log(epoch, num_epochs, epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    print(
        f"[{exp_name}] "
        f"Epoch [{epoch}/{num_epochs}] | "
        f"Time: {epoch_time:.2f}s | "
        f"Train PSNR: {train_psnr:.2f} | "
        f"Val PSNR: {val_psnr:.2f} | "
        f"Val SSIM: {val_ssim:.4f}"
    )


def validation(net, val_loader, device, exp_name):
    net.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_image, gt, _ = batch
            input_image = input_image.to(device)
            gt = gt.to(device)

            pred = net(input_image).clamp(0, 1)

            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()

            for i in range(pred_np.shape[0]):
                psnr_list.append(psnr(gt_np[i], pred_np[i], data_range=1.0))
                ssim_list.append(
                    ssim(
                        gt_np[i].transpose(1, 2, 0),
                        pred_np[i].transpose(1, 2, 0),
                        channel_axis=2,
                        data_range=1.0
                    )
                )

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)

    return mean_psnr, mean_ssim


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5, step=50):
    if epoch > 0 and epoch % step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

def total_variation_loss(mask):
    return torch.mean(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:])) + \
        torch.mean(torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]))
