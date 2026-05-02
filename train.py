import argparse
import copy
import json
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import RecordUtil, build_dataset, build_model, test, train, prepare_datasets


def default_argument_parser():
    parser = argparse.ArgumentParser(description="image-super-resolution")
    parser.add_argument('--train-file', default=4, type=int)
    parser.add_argument('--eval-file', default=4, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--f', default=5, type=int)
    parser.add_argument('--model-dir', default='./model', type=str)
    parser.add_argument('--save-history', default='./model/train_history.json', type=str,
                        help='path to save training history JSON')
    return parser


def main(args):
    args.train_file = f'./datasets/91-image_x{args.train_file}.h5'
    args.eval_file = f'./datasets/Set5_x{args.eval_file}.h5'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    model = build_model(args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.MSELoss()
    train_dataloader, test_dataloader = build_dataset(args)

    EPOCH = args.epoch
    test_frequency = args.f

    epoch_loss = RecordUtil()
    epoch_psnr = RecordUtil()
    best_weights = copy.deepcopy(model.state_dict())
    best_psnr = 0
    best_epoch = 0

    history = {
        'config': {
            'train_file': args.train_file,
            'eval_file': args.eval_file,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epoch': args.epoch,
            'test_frequency': args.f,
        },
        'train_loss': [],
        'test_psnr': [],
        'test_epoch': [],
    }

    for epoch in range(EPOCH):
        train(args, model, train_dataloader, optimizer, loss_function, epoch_loss, device, epoch)
        history['train_loss'].append(epoch_loss.val[-1])

        if epoch % test_frequency == 0:
            PSNR = test(model, test_dataloader, epoch_psnr, device, epoch)
            history['test_psnr'].append(float(PSNR))
            history['test_epoch'].append(epoch)
            if PSNR > best_psnr:
                best_epoch = epoch
                best_psnr = PSNR
                best_weights = copy.deepcopy(model.state_dict())

    torch.save(best_weights, f'{args.model_dir}/best.pth')

    history['best_epoch'] = best_epoch
    history['best_psnr'] = float(best_psnr)

    os.makedirs(os.path.dirname(args.save_history) or '.', exist_ok=True)
    with open(args.save_history, 'w') as f:
        json.dump(history, f, indent=2)

    print('\n\n')
    print(f'best epoch = {best_epoch}')
    print(f'best psnr = {best_psnr}')
    print(f'best model weights was saved in {args.model_dir}/best.pth')
    print(f'training history saved to {args.save_history}')
    print('-------------over-------------')

    return history


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)