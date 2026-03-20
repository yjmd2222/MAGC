# modified from https://github.com/modyu-liu/FaceMe/blob/main/utils/degradation.py
import numpy as np
import torch

try:
    from torchvision.transforms.functional import rgb_to_grayscale
except ImportError:
    from torchvision.transforms._functional_tensor import rgb_to_grayscale


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    """Generate Gaussian noise for a batch of images."""
    b, _, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(b, 1, 1, 1)

    if isinstance(gray_noise, (float, int)):
        gray_noise = img.new_full((b, 1, 1, 1), float(gray_noise))
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)

    use_gray = torch.sum(gray_noise) > 0
    if use_gray:
        noise_gray = torch.randn((b, 1, h, w), dtype=img.dtype, device=img.device) * sigma / 255.0

    noise = torch.randn_like(img) * sigma / 255.0
    if use_gray:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise


def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    sigma = sigma * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)


def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0
    return out


def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate Poisson noise for a batch of images."""
    b, _, h, w = img.size()

    if isinstance(gray_noise, (float, int)):
        gray_noise = img.new_full((b, 1, 1, 1), float(gray_noise))
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)

    use_gray = torch.sum(gray_noise) > 0
    if use_gray:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.0
        vals_list = [len(torch.unique(img_gray[i])) for i in range(b)]
        vals = [2 ** np.ceil(np.log2(v)) for v in vals_list]
        vals = img_gray.new_tensor(vals).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = (out - img_gray).expand(b, 3, h, w)

    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.0
    vals_list = [len(torch.unique(img[i])) for i in range(b)]
    vals = [2 ** np.ceil(np.log2(v)) for v in vals_list]
    vals = img.new_tensor(vals).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img

    if use_gray:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise

    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale


def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    scale = scale * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)


def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0
    return out
