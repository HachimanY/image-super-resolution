
from model import SRCNN
from torch.utils.data.dataloader import DataLoader
from datasets import TrainDataset, EvalDataset
import numpy as np
import torch
import numpy as np
import PIL.Image as pil_image
import h5py
import glob
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prepare_datasets(args):
    train_set_pretreatment(args)
    test_set_pretreatment(args)


def train_set_pretreatment(args):
    h5_file = h5py.File(args.train_file, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def test_set_pretreatment(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()

def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))



def build_model(args,device):
    model = SRCNN().to(device)
    return model

def build_dataset(args):

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    return train_dataloader,eval_dataloader

class RecordUtil(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = []
        self.avg = 0.0

    def update(self, val):
        self.val.append(val)
        self.avg = np.mean(self.val)

def train(args,model,train_dataloader,optimizer,loss_function,epoch_loss,device,epoch):
    
    model.train()
    total_loss = 0
    for idx,data in enumerate(train_dataloader):
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_function(output,labels)

        total_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss.update(total_loss)

    #torch.save(model.state_dict(), f'{args.model_dir}/epoch_{epoch}.pth')
    print(f'-----finish training of epoch {epoch}-----')

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def test(model,test_dataloader,epoch_psnr,device,epoch):    

    print('\n-----start test-----\n')
    model.eval()
    PSNR = []
    
    for idx,data in enumerate(test_dataloader):
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(images).clamp(0.0, 1.0)
        PSNR.append(calc_psnr(preds, labels).cpu())

    PSNR = np.mean(PSNR)
    epoch_psnr.update(PSNR)
    print(f' PSNR = {PSNR} of epoch {epoch}')
    print('\n-----end test-----\n')
    return PSNR


def plot_training_curves(history_path, save_path=None):
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['test_epoch'], history['test_psnr'], color='green', marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Test PSNR')
    ax2.grid(True, alpha=0.3)

    config = history.get('config', {})
    base = os.path.basename(history_path)
    exp_id = base[4:].split('.')[0] if base.startswith('exp_') else ''
    prefix = f'Exp {exp_id} | ' if exp_id != '' else ''
    fig.suptitle(f"{prefix}lr={config.get('lr', '?')}, batch={config.get('batch_size', '?')}, "
                 f"best_psnr={history.get('best_psnr', '?'):.2f}dB @ epoch {history.get('best_epoch', '?')}",
                 fontsize=11)
    plt.tight_layout()

    if save_path is None:
        save_path = history_path.replace('.json', '_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {save_path}')
    return save_path


def generate_comparison_image(lr_image_path, sr_image, scale, save_path=None):
    lr_img = pil_image.open(lr_image_path).convert('RGB')

    w = (lr_img.width // scale) * scale
    h = (lr_img.height // scale) * scale
    lr_img = lr_img.resize((w, h), resample=pil_image.BICUBIC)

    lr_small = lr_img.resize((w // scale, h // scale), resample=pil_image.BICUBIC)
    bicubic_img = lr_small.resize((w, h), resample=pil_image.BICUBIC)

    if isinstance(sr_image, np.ndarray):
        sr_img = pil_image.fromarray(sr_image)
    else:
        sr_img = sr_image

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(lr_small)
    axes[0].set_title(f'Low Resolution ({lr_small.width}x{lr_small.height})')
    axes[0].axis('off')

    axes[1].imshow(bicubic_img)
    axes[1].set_title(f'Bicubic Interpolation ({bicubic_img.width}x{bicubic_img.height})')
    axes[1].axis('off')

    axes[2].imshow(sr_img)
    axes[2].set_title(f'SRCNN ({sr_img.width}x{sr_img.height})')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path is None:
        base, ext = os.path.splitext(lr_image_path)
        save_path = f'{base}_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Comparison image saved to {save_path}')
    return save_path


def compare_experiments(history_paths, save_path='./experiment_comparison.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    results = []
    for hp in history_paths:
        with open(hp, 'r') as f:
            h = json.load(f)
        config = h.get('config', {})
        base = os.path.basename(hp)
        exp_id = ''
        if base.startswith('exp_'):
            exp_id = base[4:].split('.')[0]
        prefix = f'Exp {exp_id}: ' if exp_id != '' else ''
        label = f"{prefix}lr={config.get('lr')}, bs={config.get('batch_size')}"
        results.append({
            'label': label,
            'best_psnr': h.get('best_psnr', 0),
            'best_epoch': h.get('best_epoch', 0),
            'loss': h['train_loss'],
            'test_epoch': h['test_epoch'],
            'test_psnr': h['test_psnr'],
        })

    for r in results:
        ax1.plot(r['loss'], label=r['label'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss Comparison')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for r in results:
        ax2.plot(r['test_epoch'], r['test_psnr'], marker='o', markersize=3, label=r['label'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Test PSNR Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Experiment comparison saved to {save_path}')

    print('\n' + '=' * 60)
    print(f'{"Label":<40} {"Best PSNR":>10} {"Best Epoch":>11}')
    print('=' * 60)
    for r in sorted(results, key=lambda x: x['best_psnr'], reverse=True):
        print(f'{r["label"]:<40} {r["best_psnr"]:>10.2f} {r["best_epoch"]:>11}')
    print('=' * 60)

    return save_path

