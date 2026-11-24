"""Time-conditioned U-Net and cosine noise schedule for DDPM training."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=t.device) * -scale)
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(
            batch, 3, self.heads, channels // self.heads, height * width
        ).unbind(1)
        q, k, v = (item.transpose(-2, -1) for item in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-2, -1).reshape(batch, channels, height, width)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class DiffusionUNet(nn.Module):
    def __init__(self, image_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        time_dim = c * 4
        self.time_mlp = nn.Sequential(nn.Linear(c, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.stem = nn.Conv2d(image_channels, c, 3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.downsample1 = Downsample(c, 2 * c)
        self.down2 = ResBlock(2 * c, 2 * c, time_dim)
        self.downsample2 = Downsample(2 * c, 4 * c)
        self.down3 = ResBlock(4 * c, 4 * c, time_dim)
        self.downsample3 = Downsample(4 * c, 4 * c)
        self.down4 = ResBlock(4 * c, 4 * c, time_dim)
        self.attn4 = AttentionBlock(4 * c)
        self.downsample4 = Downsample(4 * c, 4 * c)
        self.mid1 = ResBlock(4 * c, 4 * c, time_dim)
        self.mid_attn = AttentionBlock(4 * c)
        self.mid2 = ResBlock(4 * c, 4 * c, time_dim)
        self.upsample4 = Upsample(4 * c, 4 * c)
        self.up4 = ResBlock(8 * c, 4 * c, time_dim)
        self.attn_up4 = AttentionBlock(4 * c)
        self.upsample3 = Upsample(4 * c, 4 * c)
        self.up3 = ResBlock(8 * c, 4 * c, time_dim)
        self.upsample2 = Upsample(4 * c, 2 * c)
        self.up2 = ResBlock(4 * c, 2 * c, time_dim)
        self.upsample1 = Upsample(2 * c, c)
        self.up1 = ResBlock(2 * c, c, time_dim)
        self.out_norm = nn.GroupNorm(32, c)
        self.out_conv = nn.Conv2d(c, image_channels, 3, padding=1)

    def forward(self, x, t):
        time = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features))
        h1 = self.down1(self.stem(x), time)
        h2 = self.down2(self.downsample1(h1), time)
        h3 = self.down3(self.downsample2(h2), time)
        h4 = self.attn4(self.down4(self.downsample3(h3), time))
        h = self.mid2(self.mid_attn(self.mid1(self.downsample4(h4), time)), time)
        h = self.attn_up4(self.up4(torch.cat((self.upsample4(h), h4), dim=1), time))
        h = self.up3(torch.cat((self.upsample3(h), h3), dim=1), time)
        h = self.up2(torch.cat((self.upsample2(h), h2), dim=1), time)
        h = self.up1(torch.cat((self.upsample1(h), h1), dim=1), time)
        return self.out_conv(F.silu(self.out_norm(h)))


def cosine_alpha_bar(timesteps, device):
    steps = torch.arange(timesteps + 1, device=device, dtype=torch.float64)
    offset = 0.008
    values = torch.cos(((steps / timesteps + offset) / (1 + offset)) * math.pi / 2).square()
    values = values / values[0]
    betas = 1 - values[1:] / values[:-1]
    betas = betas.clamp(1e-5, 0.999).float()
    return torch.cumprod(1 - betas, dim=0)


@torch.no_grad()
def ddim_sample(model, count, image_size, alpha_bar, device, steps=50, seed=None):
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(count, 3, image_size, image_size, device=device, generator=generator)
    else:
        x = torch.randn(count, 3, image_size, image_size, device=device)
    times = torch.linspace(len(alpha_bar) - 1, 0, steps, device=device).long().unique_consecutive()
    for index, timestep in enumerate(times):
        t = torch.full((count,), int(timestep), device=device, dtype=torch.long)
        alpha = alpha_bar[timestep]
        alpha_prev = alpha_bar[times[index + 1]] if index + 1 < len(times) else alpha.new_tensor(1.0)
        predicted_noise = model(x, t)
        predicted_clean = (x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        predicted_clean = predicted_clean.clamp(-1, 1)
        x = alpha_prev.sqrt() * predicted_clean + (1 - alpha_prev).sqrt() * predicted_noise
    return x.clamp(-1, 1)
