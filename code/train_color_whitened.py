"""Train a DDPM after whitening RGB channel covariance."""

import argparse
import copy
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from data import HistopathologyDataset, image_paths
from model import DiffusionUNet, cosine_alpha_bar


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_color_transform(path, device):
    values = json.loads(Path(path).read_text())
    values['mean'] = torch.tensor(values['mean'], device=device, dtype=torch.float32)
    values['forward_matrix'] = torch.tensor(values['forward_matrix'], device=device, dtype=torch.float32)
    values['inverse_matrix'] = torch.tensor(values['inverse_matrix'], device=device, dtype=torch.float32)
    return values


class WhitenedDataset(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.mean = transform['mean'].cpu()
        self.forward = transform['forward_matrix'].cpu()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = self.base[index]
        centered = image - self.mean[:, None, None]
        return torch.einsum('ij,jhw->ihw', self.forward, centered)


def inverse_color_transform(image, transform):
    return torch.einsum('ij,bjhw->bihw', transform['inverse_matrix'], image) + transform['mean'][None, :, None, None]


@torch.no_grad()
def sample(model, count, image_size, alpha_bar, device, steps, clip_value, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn(count, 3, image_size, image_size, device=device, generator=generator)
    times = torch.linspace(len(alpha_bar) - 1, 0, steps, device=device).long().unique_consecutive()
    for index, timestep in enumerate(times):
        t = torch.full((count,), int(timestep), device=device, dtype=torch.long)
        alpha = alpha_bar[timestep]
        alpha_prev = alpha_bar[times[index + 1]] if index + 1 < len(times) else alpha.new_tensor(1.0)
        noise = model(image, t)
        clean = (image - (1 - alpha).sqrt() * noise) / alpha.sqrt()
        clean = clean.clamp(-clip_value, clip_value)
        image = alpha_prev.sqrt() * clean + (1 - alpha_prev).sqrt() * noise
    return image.clamp(-clip_value, clip_value)


def save_loss_curve(csv_path, output_path, selected_epoch):
    rows = list(csv.DictReader(Path(csv_path).open()))
    epochs = [int(row['epoch']) for row in rows]
    losses = [float(row['loss']) for row in rows]
    width, height = 900, 460
    left, top, right, bottom = 86, 58, 28, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    low, high = min(losses), max(losses)
    pad = max((high - low) * 0.08, 0.005)
    low, high = low - pad, high + pad
    points = []
    for epoch, loss in zip(epochs, losses):
        x = left + (epoch - 1) / max(epochs[-1] - 1, 1) * plot_width
        y = top + (high - loss) / (high - low) * plot_height
        points.append(f'{x:.1f},{y:.1f}')
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="Arial" font-size="20">RGB-whitened DDPM training loss</text>',
    ]
    for tick in range(6):
        value = low + tick * (high - low) / 5
        y = top + (5 - tick) * plot_height / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{value:.3f}</text>')
    for epoch in range(0, epochs[-1] + 1, 20):
        x = left + epoch / epochs[-1] * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 23}" text-anchor="middle" font-family="Arial" font-size="13">{epoch}</text>')
    selected_x = left + (selected_epoch - 1) / max(epochs[-1] - 1, 1) * plot_width
    parts.extend([
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333"/>',
        f'<line x1="{selected_x:.1f}" y1="{top}" x2="{selected_x:.1f}" y2="{top + plot_height}" stroke="#b23a48" stroke-dasharray="5,5"/>',
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#315b8a" stroke-width="2"/>',
        f'<text x="{selected_x + 6:.1f}" y="{top + 16}" font-family="Arial" font-size="12" fill="#b23a48">selected epoch {selected_epoch}</text>',
        f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="15">Epoch</text>',
        f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2})" font-family="Arial" font-size="15">Noise prediction MSE</text>',
        '</svg>',
    ])
    Path(output_path).write_text('\n'.join(parts) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='outputs/ddpm128_rgb_whitened')
    parser.add_argument('--color-transform', default='outputs/ddpm128_rgb_whitened/color_transform.json')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--sampling-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--selected-epoch', type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output)
    checkpoints = output / 'checkpoints'
    samples = output / 'samples'
    checkpoints.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)

    paths = image_paths(args.data)
    if len(paths) != 5179:
        raise ValueError(f'Expected 5,179 assignment images, found {len(paths)}')
    transform = load_color_transform(args.color_transform, device)
    transform_copy = output / 'color_transform.json'
    if Path(args.color_transform).resolve() != transform_copy.resolve():
        shutil.copyfile(args.color_transform, transform_copy)

    base = HistopathologyDataset(paths, args.image_size, augment=True)
    dataset = WhitenedDataset(base, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )
    model = DiffusionUNet(base_channels=args.base_channels).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not use_bf16)
    alpha_bar = cosine_alpha_bar(args.timesteps, device)
    config = {
        'image_size': args.image_size, 'base_channels': args.base_channels,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'ema_decay': args.ema_decay,
        'timesteps': args.timesteps, 'seed': args.seed, 'images': len(paths),
        'precision': 'bf16' if use_bf16 else ('fp16' if device.type == 'cuda' else 'fp32'),
        'objective': 'epsilon MSE', 'noise_schedule': 'cosine',
        'color_transform': 'mean-centered RGB covariance whitening',
        'color_transform_file': 'color_transform.json',
    }
    (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    log_path = output / 'training.csv'
    with log_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'loss', 'learning_rate', 'seconds'])
        writer.writeheader()
        total_steps = args.epochs * len(loader)
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            started = time.time()
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                fraction = global_step / max(total_steps, 1)
                warmup = min(1.0, (global_step + 1) / 500)
                lr = args.learning_rate * warmup * 0.5 * (1 + math.cos(math.pi * min(fraction, 1.0)))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                timestep = torch.randint(0, args.timesteps, (batch.size(0),), device=device)
                noise = torch.randn_like(batch)
                alpha = alpha_bar[timestep][:, None, None, None]
                noisy = alpha.sqrt() * batch + (1 - alpha).sqrt() * noise
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                    predicted = model(noisy, timestep)
                    loss = nn.functional.mse_loss(predicted.float(), noise.float())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    for ema_param, param in zip(ema.parameters(), model.parameters()):
                        ema_param.lerp_(param, 1 - args.ema_decay)
                total_loss += float(loss.detach())
                global_step += 1

            row = {
                'epoch': epoch, 'loss': total_loss / len(loader),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'seconds': round(time.time() - started, 1),
            }
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if epoch % 5 == 0:
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'ema': ema.state_dict(), 'config': config}, checkpoints / 'latest.pt')
            if epoch % 25 == 0 or epoch == args.epochs:
                state = {'epoch': epoch, 'model': model.state_dict(), 'ema': ema.state_dict(), 'config': config}
                torch.save(state, checkpoints / f'epoch_{epoch:03d}.pt')
                ema.eval()
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                    white = sample(ema, 16, args.image_size, alpha_bar, device, args.sampling_steps, transform['clip_value'], args.seed + epoch)
                rgb = inverse_color_transform(white.float(), transform).clamp(-1, 1)
                save_image((rgb + 1) / 2, samples / f'epoch_{epoch:03d}.png', nrow=4)

    save_loss_curve(log_path, output / 'loss_curve.svg', args.selected_epoch)


if __name__ == '__main__':
    main()
