import argparse
import os
import subprocess
import sys
import json

from utils import plot_training_curves, compare_experiments


DEFAULT_EXPERIMENTS = [
    {'train_file': 2, 'eval_file': 2, 'lr': 1e-4, 'batch_size': 32, 'epoch': 100, 'f': 5},
    {'train_file': 4, 'eval_file': 4, 'lr': 1e-4, 'batch_size': 32, 'epoch': 100, 'f': 5},
    {'train_file': 4, 'eval_file': 4, 'lr': 1e-3, 'batch_size': 32, 'epoch': 100, 'f': 5},
    {'train_file': 4, 'eval_file': 4, 'lr': 1e-5, 'batch_size': 32, 'epoch': 100, 'f': 5},
    {'train_file': 4, 'eval_file': 4, 'lr': 1e-4, 'batch_size': 64, 'epoch': 100, 'f': 5},
]


def run_single_experiment(exp, index, output_dir):
    history_path = os.path.join(output_dir, f'exp_{index}.json')

    cmd = [
        sys.executable, 'train.py',
        '--train-file', str(exp['train_file']),
        '--eval-file', str(exp['eval_file']),
        '--lr', str(exp['lr']),
        '--batch-size', str(exp['batch_size']),
        '--epoch', str(exp['epoch']),
        '--f', str(exp['f']),
        '--save-history', history_path,
    ]

    print(f'\n{"=" * 60}')
    print(f'Experiment {index}: lr={exp["lr"]}, batch_size={exp["batch_size"]}, '
          f'scale=x{exp["train_file"]}, epoch={exp["epoch"]}')
    print(f'{"=" * 60}\n')

    subprocess.run(cmd, check=True)

    plot_training_curves(history_path, save_path=os.path.join(output_dir, f'exp_{index}_curves.png'))

    return history_path


def plot_only(output_dir):
    """Read existing experiment JSON files and generate charts + summary."""
    import glob as glob_mod
    pattern = os.path.join(output_dir, 'exp_*.json')
    files = sorted(glob_mod.glob(pattern))
    if not files:
        print(f'No experiment JSON files found in {output_dir}')
        return

    print(f'Found {len(files)} experiment files:')
    for f in files:
        print(f'  {f}')

    # Generate individual training curves
    for hp in files:
        plot_training_curves(hp)

    # Generate comparison chart
    compare_experiments(files, save_path=os.path.join(output_dir, 'comparison.png'))

    # Print summary
    print('\n' + '=' * 70)
    print('EXPERIMENT RESULTS SUMMARY')
    print('=' * 70)
    print(f'{"#":<4} {"Config":<35} {"Best PSNR":>10} {"Best Epoch":>11}')
    print('-' * 70)
    for hp in files:
        with open(hp, 'r') as f:
            h = json.load(f)
        config = h.get('config', {})
        base = os.path.basename(hp)
        exp_id = base.replace('exp_', '').replace('.json', '')
        label = f"lr={config.get('lr')}, bs={config.get('batch_size')}"
        print(f'{exp_id:<4} {label:<35} {h["best_psnr"]:>10.2f} {h["best_epoch"]:>11}')
    print('=' * 70)


def main():
    parser = argparse.ArgumentParser(description='Run parameter comparison experiments')
    parser.add_argument('--output-dir', default='./experiments', type=str,
                        help='directory to save experiment results')
    parser.add_argument('--epochs', default=None, type=int,
                        help='override epoch count for all experiments')
    parser.add_argument('--quick', action='store_true',
                        help='run a quick set of experiments (fewer epochs)')
    parser.add_argument('--plot-only', action='store_true',
                        help='skip training, only generate charts from existing data')
    args = parser.parse_args()

    if args.plot_only:
        plot_only(args.output_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = DEFAULT_EXPERIMENTS.copy()
    if args.quick:
        for exp in experiments:
            exp['epoch'] = 20
            exp['f'] = 5
    elif args.epochs:
        for exp in experiments:
            exp['epoch'] = args.epochs

    history_paths = []
    for i, exp in enumerate(experiments):
        hp = run_single_experiment(exp, i, args.output_dir)
        history_paths.append(hp)

    compare_experiments(history_paths, save_path=os.path.join(args.output_dir, 'comparison.png'))

    summary = []
    for i, (exp, hp) in enumerate(zip(experiments, history_paths)):
        with open(hp, 'r') as f:
            h = json.load(f)
        summary.append({
            'experiment': i,
            'config': exp,
            'best_psnr': h['best_psnr'],
            'best_epoch': h['best_epoch'],
        })

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to {summary_path}')

    print('\n' + '=' * 70)
    print('EXPERIMENT RESULTS SUMMARY')
    print('=' * 70)
    print(f'{"#":<4} {"Scale":<7} {"LR":<10} {"Batch":<7} {"Best PSNR":>10} {"Best Epoch":>11}')
    print('-' * 70)
    for s in sorted(summary, key=lambda x: x['best_psnr'], reverse=True):
        c = s['config']
        print(f'{s["experiment"]:<4} x{c["train_file"]:<6} {c["lr"]:<10} {c["batch_size"]:<7} '
              f'{s["best_psnr"]:>10.2f} {s["best_epoch"]:>11}')
    print('=' * 70)


if __name__ == '__main__':
    main()
