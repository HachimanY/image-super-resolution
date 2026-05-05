"""
Generate a comparison figure showing SR results at different PSNR levels.

Usage:
  python compare_psnr.py --image <test_image> --checkpoints-dir ./model --scale 2
  python compare_psnr.py --image <test_image> --checkpoints epoch_10.pth,epoch_30.pth,epoch_50.pth,epoch_100.pth
"""

import argparse
import os
import re
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


def load_model(weights_file, device):
    model = SRCNN().to(device)
    state_dict = model.state_dict()
    loaded = torch.load(weights_file, map_location=lambda storage, loc: storage)
    for n, p in loaded.items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
    model.eval()
    return model


def sr_inference(model, image, scale, device):
    """Run SRCNN on an image, return SR output as uint8 numpy array and PSNR."""
    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

    image_np = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image_np)

    new_channels = []
    psnr_vals = []
    for i in range(3):
        y = ycbcr[..., i] / 255.0
        y_tensor = torch.from_numpy(y).to(device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            preds = model(y_tensor).clamp(0.0, 1.0)
        psnr_vals.append(calc_psnr(y_tensor, preds).cpu())
        new_channels.append(preds.mul(255.0).cpu().numpy().squeeze())

    avg_psnr = float(np.mean(psnr_vals))
    output = np.array(new_channels).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    return output, avg_psnr


def get_bicubic(image, scale):
    """Get the bicubic interpolation baseline as uint8 numpy array."""
    w = (image.width // scale) * scale
    h = (image.height // scale) * scale
    img = image.resize((w, h), resample=pil_image.BICUBIC)
    lr = img.resize((w // scale, h // scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((w, h), resample=pil_image.BICUBIC)
    return np.array(bicubic)


def get_lr(image, scale):
    """Get the low-resolution image (upscaled back via nearest for display)."""
    w = (image.width // scale) * scale
    h = (image.height // scale) * scale
    img = image.resize((w, h), resample=pil_image.BICUBIC)
    lr = img.resize((w // scale, h // scale), resample=pil_image.BICUBIC)
    return np.array(lr)


def extract_epoch(filename):
    """Extract epoch number from filename like 'epoch_30.pth'."""
    m = re.search(r'epoch_(\d+)', filename)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description='Compare SR results at different PSNR levels')
    parser.add_argument('--image', type=str, required=True, help='path to test image')
    parser.add_argument('--scale', type=int, default=2, help='upscale factor')
    parser.add_argument('--checkpoints-dir', type=str, default='./model',
                        help='directory containing epoch_*.pth checkpoints')
    parser.add_argument('--checkpoints', type=str, default='',
                        help='comma-separated checkpoint filenames (overrides --checkpoints-dir)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='output path for the comparison figure')
    parser.add_argument('--show-bicubic', action='store_true', default=True,
                        help='include bicubic baseline in comparison')
    parser.add_argument('--crop-region', type=str, default=None,
                        help='crop region "x,y,w,h" for zoomed detail comparison')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoint list
    if args.checkpoints:
        ckpt_files = [c.strip() for c in args.checkpoints.split(',')]
    else:
        ckpt_files = sorted(
            [f for f in os.listdir(args.checkpoints_dir) if f.startswith('epoch_') and f.endswith('.pth')],
            key=lambda x: extract_epoch(x) or 0
        )

    if not ckpt_files:
        print('No checkpoints found. Train with --save-epochs first, e.g.:')
        print('  python train.py --epoch 100 --save-epochs 10,30,50,100')
        return

    print(f'Found {len(ckpt_files)} checkpoints: {ckpt_files}')

    image = pil_image.open(args.image).convert('RGB')
    lr_img = get_lr(image, args.scale)
    bicubic_img = get_bicubic(image, args.scale)

    # Run SR with each checkpoint
    results = []
    for ckpt_name in ckpt_files:
        ckpt_path = ckpt_name if os.path.isabs(ckpt_name) else os.path.join(args.checkpoints_dir, ckpt_name)
        epoch_num = extract_epoch(ckpt_name)
        model = load_model(ckpt_path, device)
        sr_img, psnr = sr_inference(model, image, args.scale, device)
        label = f'Epoch {epoch_num}' if epoch_num is not None else ckpt_name
        results.append({'label': label, 'epoch': epoch_num, 'psnr': psnr, 'img': sr_img})
        print(f'{label}: PSNR = {psnr:.2f} dB')

    # Sort by PSNR
    results.sort(key=lambda x: x['psnr'])

    # --- Plot ---
    n_cols = len(results)
    if args.show_bicubic:
        n_cols += 1  # add bicubic column

    fig_height = 6
    fig_width = max(5 * n_cols, 12)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))
    if n_cols == 1:
        axes = [axes]

    col = 0
    if args.show_bicubic:
        bicubic_psnr = calc_psnr(
            torch.from_numpy(bicubic_img.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0),
            torch.from_numpy(np.array(image.resize(
                (bicubic_img.shape[1], bicubic_img.shape[0]),
                resample=pil_image.BICUBIC
            )).astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        ).item()
        axes[col].imshow(bicubic_img)
        axes[col].set_title(f'Bicubic\n(PSNR baseline)', fontsize=11)
        axes[col].axis('off')
        col += 1

    for r in results:
        axes[col].imshow(r['img'])
        axes[col].set_title(f"{r['label']}\nPSNR: {r['psnr']:.2f} dB", fontsize=11)
        axes[col].axis('off')
        col += 1

    fig.suptitle(f'Super-Resolution Comparison (x{args.scale})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if args.save_path is None:
        base = os.path.splitext(args.image)[0]
        args.save_path = f'{base}_psnr_comparison.png'
    plt.savefig(args.save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nComparison figure saved to {args.save_path}')

    # Optional: crop detail comparison
    if args.crop_region:
        x, y, cw, ch = [int(v) for v in args.crop_region.split(',')]
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(fig_width, fig_height))
        if n_cols == 1:
            axes2 = [axes2]

        col = 0
        if args.show_bicubic:
            crop = bicubic_img[y:y+ch, x:x+cw]
            axes2[col].imshow(crop)
            axes2[col].set_title(f'Bicubic (detail)', fontsize=11)
            axes2[col].axis('off')
            col += 1

        for r in results:
            crop = r['img'][y:y+ch, x:x+cw]
            axes2[col].imshow(crop)
            axes2[col].set_title(f"{r['label']}\nPSNR: {r['psnr']:.2f} dB", fontsize=11)
            axes2[col].axis('off')
            col += 1

        fig2.suptitle(f'Detail Comparison (x{args.scale})', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        detail_path = args.save_path.replace('.png', '_detail.png')
        plt.savefig(detail_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Detail figure saved to {detail_path}')


if __name__ == '__main__':
    main()
