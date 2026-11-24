"""Evaluate the RGB-whitened DDPM with PyTorch-Ignite metrics."""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ignite.metrics import FID, InceptionScore
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.utils import save_image

sys.path.insert(0, os.environ.get('PROJECT_CODE_DIR', str(Path(__file__).resolve().parent)))
from data import HistopathologyDataset, image_paths
from model import DiffusionUNet, cosine_alpha_bar


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
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((width // 2 - 145, 18), 'RGB-whitened DDPM training loss', fill='#222222', font=font)
    for tick in range(6):
        value = low + tick * (high - low) / 5
        y = top + (5 - tick) * plot_height / 5
        draw.line((left, y, left + plot_width, y), fill='#dddddd', width=1)
        draw.text((left - 55, y - 6), f'{value:.3f}', fill='#333333', font=font)
    for epoch in range(0, epochs[-1] + 1, 20):
        x = left + epoch / epochs[-1] * plot_width
        draw.line((x, top, x, top + plot_height), fill='#eeeeee', width=1)
        draw.text((x - 8, top + plot_height + 10), str(epoch), fill='#333333', font=font)
    draw.line((left, top, left, top + plot_height), fill='#333333', width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill='#333333', width=2)
    points = []
    for epoch, loss in zip(epochs, losses):
        x = left + (epoch - 1) / max(epochs[-1] - 1, 1) * plot_width
        y = top + (high - loss) / (high - low) * plot_height
        points.append((x, y))
    draw.line(points, fill='#315b8a', width=3)
    selected_x = left + (selected_epoch - 1) / max(epochs[-1] - 1, 1) * plot_width
    for y in range(top, top + plot_height, 12):
        draw.line((selected_x, y, selected_x, min(y + 6, top + plot_height)), fill='#b23a48', width=2)
    draw.text((left + plot_width // 2 - 16, height - 35), 'Epoch', fill='#222222', font=font)
    image.save(output_path)


def inverse_color_transform(image, transform):
    matrix = torch.tensor(transform['inverse_matrix'], device=image.device, dtype=torch.float32)
    mean = torch.tensor(transform['mean'], device=image.device, dtype=torch.float32)
    return torch.einsum('ij,bjhw->bihw', matrix, image) + mean[None, :, None, None]


@torch.no_grad()
def sample(model, count, image_size, alpha_bar, device, steps, clip_value, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn(count, 3, image_size, image_size, device=device, generator=generator)
    times = torch.linspace(len(alpha_bar) - 1, 0, steps, device=device).long().unique_consecutive()
    for index, timestep in enumerate(times):
        t = torch.full((count,), int(timestep), device=device, dtype=torch.long)
        alpha = alpha_bar[timestep]
        alpha_prev = alpha_bar[times[index + 1]] if index + 1 < len(times) else alpha.new_tensor(1.0)
        predicted_noise = model(image, t)
        clean = (image - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        clean = clean.clamp(-clip_value, clip_value)
        image = alpha_prev.sqrt() * clean + (1 - alpha_prev).sqrt() * predicted_noise
    return image.clamp(-clip_value, clip_value)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--color-transform', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-samples', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sampling-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(output.parent / 'cache' / 'torch'))

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    transform = json.loads(Path(args.color_transform).read_text())
    config = state['config']
    model = DiffusionUNet(base_channels=config['base_channels']).to(device)
    model.load_state_dict(state['ema'])
    model.eval()
    alpha_bar = cosine_alpha_bar(config['timesteps'], device)

    paths = image_paths(args.data)
    if len(paths) < args.num_samples:
        raise ValueError(f'Need {args.num_samples} real images, found {len(paths)}')
    selected = random.Random(args.seed).sample(paths, args.num_samples)
    dataset = HistopathologyDataset(selected, config['image_size'], augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
    )

    fid = FID(device=device)
    inception = InceptionScore(device=device)
    saved = []
    generated = 0
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    clip_value = float(transform['clip_value'])

    for real in loader:
        count = real.size(0)
        real = real.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            white = sample(
                model, count, config['image_size'], alpha_bar, device,
                args.sampling_steps, clip_value, args.seed + generated,
            )
        fake = inverse_color_transform(white.float(), transform).clamp(-1, 1)
        fake = ((fake + 1) / 2).clamp(0, 1)
        real = ((real.float() + 1) / 2).clamp(0, 1)
        fake_inception = resize(fake, [299, 299], interpolation=InterpolationMode.BILINEAR, antialias=True)
        real_inception = resize(real, [299, 299], interpolation=InterpolationMode.BILINEAR, antialias=True)
        fid.update((fake_inception, real_inception))
        inception.update(fake_inception)
        saved_count = sum(batch.size(0) for batch in saved)
        if saved_count < 64:
            saved.append(fake[:min(count, 64 - saved_count)].cpu())
        generated += count
        print(f'Evaluated {generated}/{args.num_samples} samples', flush=True)

    is_result = inception.compute()
    if isinstance(is_result, (tuple, list)):
        is_score, is_std = map(float, is_result)
    else:
        is_score, is_std = float(is_result), None
    metrics = {
        'checkpoint_epoch': int(state['epoch']),
        'dataset': 'Kaggle rushilprajapati/data-final, Histopathology subset',
        'real_images': args.num_samples,
        'generated_images': generated,
        'real_reference': 'deterministic subset of the training collection; not an independent test split',
        'resolution': config['image_size'],
        'ddim_sampling_steps': args.sampling_steps,
        'metric': 'PyTorch-Ignite default Inception-v3 FID and Inception Score',
        'fid': float(fid.compute()),
        'inception_score': is_score,
        'inception_score_std': is_std,
        'seed': args.seed,
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    save_image(torch.cat(saved), output / 'generated_samples.png', nrow=8)
    history = Path(args.checkpoint).parents[1] / 'training.csv'
    save_loss_curve(history, output / 'loss_curve.png', int(state['epoch']))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
