import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.utils import make_grid, save_image

import csv
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend suitable for saving to files
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse

import torch.backends.cudnn as cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

os.makedirs("saved_models", exist_ok=True)


# -------- Setting Seed -------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




# ---------- Differentiable Transform Modules ----------

def color_shift(x, color_matrix, bias):
    """
    x: [B, C, D, H, W]
    color_matrix: [B, C, C]
    bias: [C]
    """
    B, C, D, H, W = x.shape
    mean = x.mean(dim=[2, 3, 4], keepdim=True)
    x_centered = x - mean

    # batched color transform: contract input channel (c) with matrix col (j)
    # color_matrix: [B, C_out, C_in] -> output channel is i
    x_trans = torch.einsum('bcdhw,bic->bidhw', x_centered, color_matrix)

    # add bias and restore mean
    x_trans = x_trans + bias.view(B, C, 1, 1, 1) + mean
    return x_trans.clamp(0, 1)


class DeterministicAvgPool3d(nn.Module):
    """Deterministic replacement for nn.AvgPool3d using unfold + mean."""
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride

    def forward(self, x):
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        # Use unfold to extract patches and compute mean (fully deterministic)
        patches = x.unfold(2, kD, sD).unfold(3, kH, sH).unfold(4, kW, sW)
        return patches.mean(dim=(-3, -2, -1))


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
            DeterministicAvgPool3d(2, stride=2),  # deterministic replacement for AvgPool3d
            nn.Conv3d(8, 10, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # dynamically determine the flattened size (10 * 4 * 4 * 4 = 640)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)
        )

        # initialize as identity affine transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,0], dtype=torch.float)
        )

        # dynamically determine the flattened size (10 * 4 * 4 * 4 = 640)
        self.fc_color = nn.Sequential(
            nn.Linear(10 * 4 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)
        )

        # initialize as identity affine transform
        self.fc_color[2].weight.data.zero_()
        self.fc_color[2].bias.data.copy_(
            torch.tensor([1,0,0, 0,1,0, 0,0,1, 0,0,0], dtype=torch.float)
        )

    @staticmethod
    def _deterministic_avg_pool3d(x, kernel_size, stride):
        """
        Deterministic replacement for F.avg_pool3d using unfold + mean.
        This avoids the non-deterministic avg_pool3d_backward_cuda.
        """
        B, C, D, H, W = x.shape
        kD, kH, kW = kernel_size
        sD, sH, sW = stride

        # Calculate output dimensions
        oD = (D - kD) // sD + 1
        oH = (H - kH) // sH + 1
        oW = (W - kW) // kW + 1

        # Use strided view to extract patches and compute mean
        # This is deterministic as it only uses basic tensor operations
        patches = x.unfold(2, kD, sD).unfold(3, kH, sH).unfold(4, kW, sW)
        # patches shape: [B, C, oD, oH, oW, kD, kH, kW]
        return patches.mean(dim=(-3, -2, -1))

    @staticmethod
    def _deterministic_adaptive_avg_pool3d(x, output_size):
        """Deterministic replacement for F.adaptive_avg_pool3d."""
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        oD, oH, oW = output_size
        kD, kH, kW = D // oD, H // oH, W // oW
        return SpatialTransformer._deterministic_avg_pool3d(x, (kD, kH, kW), (kD, kH, kW))

    @staticmethod
    def _deterministic_grid_sample_3d(input, grid):
        """
        Deterministic trilinear interpolation replacement for F.grid_sample.
        Uses basic tensor operations which have deterministic backwards.

        Args:
            input: [B, C, D, H, W] input tensor
            grid: [B, D_out, H_out, W_out, 3] sampling grid with values in [-1, 1]
        """
        B, C, D, H, W = input.shape
        _, oD, oH, oW, _ = grid.shape

        # Convert grid from [-1, 1] to [0, D-1], [0, H-1], [0, W-1]
        # grid[..., 0] is x (width), grid[..., 1] is y (height), grid[..., 2] is z (depth)
        grid_x = ((grid[..., 0] + 1) / 2) * (W - 1)
        grid_y = ((grid[..., 1] + 1) / 2) * (H - 1)
        grid_z = ((grid[..., 2] + 1) / 2) * (D - 1)

        # Get corner coordinates for trilinear interpolation
        x0 = grid_x.floor().long()
        x1 = x0 + 1
        y0 = grid_y.floor().long()
        y1 = y0 + 1
        z0 = grid_z.floor().long()
        z1 = z0 + 1

        # Clamp to valid range
        x0 = x0.clamp(0, W - 1)
        x1 = x1.clamp(0, W - 1)
        y0 = y0.clamp(0, H - 1)
        y1 = y1.clamp(0, H - 1)
        z0 = z0.clamp(0, D - 1)
        z1 = z1.clamp(0, D - 1)

        # Compute interpolation weights
        wx = grid_x - grid_x.floor()
        wy = grid_y - grid_y.floor()
        wz = grid_z - grid_z.floor()

        # Expand for broadcasting with channels
        wx = wx.unsqueeze(1)  # [B, 1, oD, oH, oW]
        wy = wy.unsqueeze(1)
        wz = wz.unsqueeze(1)

        # Gather values at 8 corners using advanced indexing
        # Create batch indices
        b_idx = torch.arange(B, device=input.device).view(B, 1, 1, 1).expand(B, oD, oH, oW)

        # Gather all 8 corners [B, C, oD, oH, oW]
        def gather_nd(t, z, y, x):
            # t: [B, C, D, H, W], indices: [B, oD, oH, oW]
            B, C, D, H, W = t.shape
            _, oD, oH, oW = z.shape
            # Flatten spatial dims for gathering
            t_flat = t.view(B, C, -1)  # [B, C, D*H*W]
            idx = z * H * W + y * W + x  # [B, oD, oH, oW]
            idx = idx.view(B, 1, -1).expand(B, C, -1)  # [B, C, oD*oH*oW]
            gathered = torch.gather(t_flat, 2, idx)  # [B, C, oD*oH*oW]
            return gathered.view(B, C, oD, oH, oW)

        # 8 corners for trilinear interpolation
        c000 = gather_nd(input, z0, y0, x0)
        c001 = gather_nd(input, z0, y0, x1)
        c010 = gather_nd(input, z0, y1, x0)
        c011 = gather_nd(input, z0, y1, x1)
        c100 = gather_nd(input, z1, y0, x0)
        c101 = gather_nd(input, z1, y0, x1)
        c110 = gather_nd(input, z1, y1, x0)
        c111 = gather_nd(input, z1, y1, x1)

        # Trilinear interpolation
        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        output = c0 * (1 - wz) + c1 * wz

        return output

    def forward(self, x):
        loc_feat = self.localization(x)
        xs = self._deterministic_adaptive_avg_pool3d(loc_feat, (4, 4, 4))
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 3, 4)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_stn = self._deterministic_grid_sample_3d(x, grid)

        p_color = self.fc_color(xs).view(-1, 4, 3)
        color_matrix = p_color[:, :3, :].clamp(-2.0, 2.0)  # Constrain color matrix
        bias = p_color[:, 3, :].clamp(-0.5, 0.5)  # Constrain bias
        x_color = color_shift(x_stn, color_matrix, bias)
        return x_color


class DoGFilter3D(nn.Module):
    def __init__(self, sigma1=1.0, sigma2=2.0, channels=3):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(sigma1))
        self.sigma2 = nn.Parameter(torch.tensor(sigma2))
        self.channels = channels

    def gaussian_kernel(self, sigma):
        # Always use odd kernel size to avoid padding warning
        size = int(2 * (3 * sigma) + 1)
        if size % 2 == 0:
            size += 1  # Ensure odd size
        size = min(size, 15)  # Cap kernel size to prevent explosion
        x = torch.arange(size).float() - size // 2
        x = x.to(sigma.device)
        g = torch.exp(-x ** 2 / (2 * sigma ** 2 + 1e-8))  # Add epsilon for stability
        g = g / (g.sum() + 1e-8)
        return g

    def forward(self, x):
        # Clamp sigma values to be positive and reasonable, handle NaN
        sigma1 = self.sigma1.clamp(0.5, 5.0)
        sigma2 = self.sigma2.clamp(0.5, 5.0)
        # Reset to default if NaN
        if torch.isnan(sigma1):
            sigma1 = torch.tensor(1.0, device=sigma1.device)
        if torch.isnan(sigma2):
            sigma2 = torch.tensor(2.0, device=sigma2.device)
        g1, g2 = self.gaussian_kernel(sigma1), self.gaussian_kernel(sigma2)
        g1_3d = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
        g2_3d = g2[:, None, None] * g2[None, :, None] * g2[None, None, :]
        g1_3d, g2_3d = g1_3d.to(x.device), g2_3d.to(x.device)
        k1 = g1_3d.expand(self.channels, 1, *g1_3d.shape)
        k2 = g2_3d.expand(self.channels, 1, *g2_3d.shape)
        # Use explicit padding to avoid warning with even kernel sizes
        # padding = kernel_size // 2 for 'same' output with odd kernels
        pad1 = g1_3d.shape[0] // 2
        pad2 = g2_3d.shape[0] // 2
        o1 = F.conv3d(x, k1, padding=pad1, groups=self.channels)
        o2 = F.conv3d(x, k2, padding=pad2, groups=self.channels)
        return o1 - o2


class BrightnessContrastAdjust(nn.Module):
    def __init__(self):
        super().__init__()
        self.brightness = nn.Parameter(torch.zeros(1))
        self.contrast = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=[2,3,4], keepdim=True)
        contrast = self.contrast.clamp(0.5, 2.0)  # Prevent extreme contrast
        brightness = self.brightness.clamp(-0.3, 0.3)  # Prevent extreme brightness
        x = (x - mean) * contrast + mean + brightness
        return x.clamp(0, 1)


class GammaAdjust(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x.clamp(min=1e-7), self.gamma.clamp(0.5, 2.0))

class GlobalColorTransform(nn.Module):
    """
    Learnable global color transformation based on whole image statistics.
    Each channel can mix linearly with others + global color bias.
    """
    def __init__(self, num_channels=3):
        super().__init__()
        # initialize as near-identity color mixing
        self.color_matrix = nn.Parameter(torch.eye(num_channels) + 0.01 * torch.randn(num_channels, num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        x_centered = x - mean
        # Constrain color matrix to prevent extreme transformations
        color_matrix = self.color_matrix.clamp(-2.0, 2.0)
        bias = self.bias.clamp(-0.5, 0.5)
        # (B, C, D, H, W) -> apply color matrix
        x_trans = torch.tensordot(x_centered, color_matrix, dims=([1],[1])).permute(0,4,1,2,3)
        x_trans = x_trans + bias.view(1, C, 1, 1, 1) + mean
        return x_trans.clamp(0, 1)


class TransformModule(nn.Module):
    def __init__(self, lambda_residual=0.5, disable_stn=False, disable_intensity=False, disable_color=False):
        super().__init__()

        self.stn = SpatialTransformer()
        self.dog = DoGFilter3D()
        self.bc = BrightnessContrastAdjust()
        self.gamma = GammaAdjust()
        self.color = GlobalColorTransform()

        self.disable_stn = disable_stn
        self.disable_intensity = disable_intensity
        self.disable_color = disable_color

        # Fixed lambda for extent of operation, each operation to be residual (1 - lambda)
        # Store as buffer (not parameter) so it's part of state_dict but not trainable
        self.register_buffer('lambda_stn', torch.tensor(lambda_residual))
        self.register_buffer('lambda_dog', torch.tensor(lambda_residual))
        self.register_buffer('lambda_bc', torch.tensor(lambda_residual))
        self.register_buffer('lambda_gamma', torch.tensor(lambda_residual))
        self.register_buffer('lambda_color', torch.tensor(lambda_residual))

    def forward(self, x):
        # x = lambda * x + (1 - lambda) * transform(x)
        if not self.disable_stn:
            x_orig = x
            x = self.lambda_stn.clamp(0, 1) * x + (1 - self.lambda_stn.clamp(0, 1)) * self.stn(x)

        if not self.disable_intensity:
            x_orig = x
            x = self.lambda_dog.clamp(0, 1) * x + (1 - self.lambda_dog.clamp(0, 1)) * self.dog(x)

            x_orig = x
            x = self.lambda_bc.clamp(0, 1) * x + (1 - self.lambda_bc.clamp(0, 1)) * self.bc(x)

            x_orig = x
            x = self.lambda_gamma.clamp(0, 1) * x + (1 - self.lambda_gamma.clamp(0, 1)) * self.gamma(x)

        if not self.disable_color:
            x_orig = x
            x = self.lambda_color.clamp(0, 1) * x + (1 - self.lambda_color.clamp(0, 1)) * self.color(x)

        return x

# ---------- Shared Backbone + Two Heads (Video Swin Transformer) ----------

class DualHeadSwin3D(nn.Module):
    def __init__(self, num_classes_S, num_classes_R):
        super().__init__()
        backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        print("Loaded swin3d_t with Kinetics-400 pretrained weights")
        in_feat = backbone.head.in_features  # 768
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.head_S = nn.Linear(in_feat, num_classes_S)
        self.head_R = nn.Linear(in_feat, num_classes_R)

    def forward(self, x, domain='S', return_features=False):
        feats = self.backbone(x)
        logits = self.head_S(feats) if domain == 'S' else self.head_R(feats)
        return (logits, feats) if return_features else logits


# ---------- MMD Loss ----------

import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def init(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).init()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

mmd = MMD_loss()

def mmd_loss(x, y):
    """Compute unbiased MMD loss with multi-scale RBF kernels."""
    Bx, By = x.size(0), y.size(0)
    assert Bx == By, "MMD requires equal batch sizes; use random sampling if unequal."

    # Pairwise squared Euclidean distances
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * xy

    # Multi-scale Gaussian kernel
    sigmas = torch.tensor([1e-6, 1e-3, 1, 3, 5, 10], device=x.device)
    beta = 1. / (2. * sigmas)
    kernels_xx = [torch.exp(-b * dxx) for b in beta]
    kernels_yy = [torch.exp(-b * dyy) for b in beta]
    kernels_xy = [torch.exp(-b * dxy) for b in beta]

    kxx = sum(kernels_xx) / len(kernels_xx)
    kyy = sum(kernels_yy) / len(kernels_yy)
    kxy = sum(kernels_xy) / len(kernels_xy)

    mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return mmd

def coral_loss(source, target):
    """Compute CORAL covariance alignment loss."""
    d = source.size(1)
    ns = source.size(0)
    nt = target.size(0)

    # source covariance
    xm_s = source - source.mean(0, keepdim=True)
    xc_s = torch.matmul(xm_s.t(), xm_s) / (ns - 1)

    # target covariance
    xm_t = target - target.mean(0, keepdim=True)
    xc_t = torch.matmul(xm_t.t(), xm_t) / (nt - 1)

    loss = torch.sum((xc_s - xc_t) ** 2) / (4 * d * d)
    return loss

# ---------- Visualization ----------

def visualize_batch(original_S, transformed_S, reference_R, out_dir, epoch, step, max_imgs=8):
    os.makedirs(out_dir, exist_ok=True)
    def slice_grid(vols):
        mid = vols.shape[2] // 2
        imgs = vols[:, :, mid, :, :]
        return make_grid(imgs[:max_imgs], nrow=4, normalize=True, scale_each=True)
    save_image(slice_grid(original_S), f"{out_dir}/e{epoch:03d}_s{step:04d}_S.png")
    save_image(slice_grid(transformed_S), f"{out_dir}/e{epoch:03d}_s{step:04d}_Strans.png")
    save_image(slice_grid(reference_R), f"{out_dir}/e{epoch:03d}_s{step:04d}_R.png")

def update_loss_plot(loss_history, out_path="loss_curve.png"):
    """
    Draw multiple loss curves and update the figure after each epoch.

    loss_history: dict
        e.g. {
            "total": [0.52, 0.48, 0.44, ...],
            "cls_S": [0.22, 0.19, 0.18, ...],
            "cls_R": [0.20, 0.17, 0.16, ...],
            "mmd": [0.10, 0.12, 0.10, ...],
        }
    """
    plt.figure(figsize=(7, 5))

    for key, values in loss_history.items():
        plt.plot(range(1, len(values)+1), values, marker='o', label=key)

    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- Training Loop ----------

def train_joint_alignment(
    model, transform_S, dataloader_S, dataloader_R, optimizer, scheduler, device,
    lambda_mmd=0.5, loss_type='mmd', viz_dir="visuals", viz_interval=200, epoch=0
):
    model.train(); transform_S.train()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0
    # x = \lambda x + (1-\lambda) self.stn(x)
    total_image_num = 0
    losses = {}
    for step, ((s_batch, s_label), (r_batch, r_label)) in enumerate(zip(dataloader_S, dataloader_R)):
        s_batch, r_batch = s_batch.to(device), r_batch.to(device)
        s_label, r_label = s_label.to(device), r_label.to(device)

        # Transform and forward
        s_trans = transform_S(s_batch)
        model.eval()
        logits_s, feat_s = model(s_trans, domain='S', return_features=True)
        model.train()
        logits_r, feat_r = model(r_batch, domain='R', return_features=True)

        # Loss
        if lambda_mmd > 1e-6:
            loss_s = 0.5 * ce_loss(logits_s, s_label)
            loss_r = ce_loss(logits_r, r_label)
            loss_cls = loss_s + loss_r
            if feat_s.shape == feat_r.shape:
                if loss_type == 'mmd':
                    loss_align = lambda_mmd * mmd_loss(feat_s, feat_r)
                elif loss_type == 'coral':
                    loss_align = lambda_mmd * coral_loss(feat_s, feat_r)
                else:
                    loss_align = 0
                loss_total = loss_cls + loss_align
            else:
                loss_align = 0
                loss_total = loss_cls
        else:
            loss_total = ce_loss(logits_r, r_label)

        # Check for NaN in loss before backprop
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            print(f"[WARNING] NaN/Inf detected at epoch {epoch}, step {step}")
            print(f"  loss_total: {loss_total.item()}")
            if lambda_mmd > 1e-6:
                print(f"  loss_s: {loss_s.item()}, loss_r: {loss_r.item()}, loss_align: {loss_align.item() if isinstance(loss_align, torch.Tensor) else loss_align}")
            print(f"  feat_s stats: min={feat_s.min().item():.4f}, max={feat_s.max().item():.4f}, mean={feat_s.mean().item():.4f}")
            print(f"  feat_r stats: min={feat_r.min().item():.4f}, max={feat_r.max().item():.4f}, mean={feat_r.mean().item():.4f}")
            print(f"  s_trans stats: min={s_trans.min().item():.4f}, max={s_trans.max().item():.4f}, mean={s_trans.mean().item():.4f}")
            continue  # Skip this batch

        optimizer.zero_grad()
        loss_total.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for name, param in list(model.named_parameters()) + list(transform_S.named_parameters()):
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"[WARNING] NaN/Inf gradient in {name}")
                has_nan_grad = True

        if has_nan_grad:
            optimizer.zero_grad()  # Clear bad gradients
            continue

        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(transform_S.parameters()), max_norm=1.0)
        optimizer.step()
        total_loss += loss_total.item()

        if step % viz_interval == 0:
            visualize_batch(s_batch.cpu(), s_trans.cpu(), r_batch.cpu(), viz_dir, epoch, step)

    scheduler.step()
    return total_loss / len(dataloader_R)


# ---------- Evaluation Function ----------

@torch.no_grad()
def evaluate(model, dataloader_S_val, dataloader_R_val, device):
    model.eval()
    correct_S = total_S = 0
    correct_R = total_R = 0

    for x, y in dataloader_S_val:
        x, y = x.to(device), y.to(device)
        preds = model(x, domain='S').argmax(dim=1)
        correct_S += (preds == y).sum().item()
        total_S += y.size(0)

    for x, y in dataloader_R_val:
        x, y = x.to(device), y.to(device)
        preds = model(x, domain='R').argmax(dim=1)
        correct_R += (preds == y).sum().item()
        total_R += y.size(0)

    acc_S = 100 * correct_S / total_S if total_S > 0 else 0
    acc_R = 100 * correct_R / total_R if total_R > 0 else 0
    return acc_S, acc_R



# ---------- Example Main Script ----------

import dataset_target_nshot
import dataset_source_full
from torch.optim import SGD, lr_scheduler, AdamW

# Override batch size for Swin3D (higher memory usage than ResNet3D)
import config_simulated_c10
config_simulated_c10.BATCH_SIZE = 4
dataset_target_nshot.BATCH_SIZE = 4
dataset_source_full.BATCH_SIZE = 4

def print_module_parameters(module, indent=0):
    prefix = " " * indent
    for name, param in module.named_parameters(recurse=False):
        print(f"{prefix}{name:30s}  shape={tuple(param.shape)}  requires_grad={param.requires_grad}")
    for child_name, child in module.named_children():
        print(f"{prefix}<{child_name}>")
        print_module_parameters(child, indent + 4)


def main_exp(args):
    set_seed(args.seed)

    if torch.cuda.is_available():
        if args.cuda_device >= torch.cuda.device_count():
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import target config dynamically and override batch size
    if args.dataset == "qiang":
        import config_Qiang as target_config
    else:
        import config_Noble as target_config
    target_config.BATCH_SIZE = 4

    dataloader_R, _, val_R = dataset_target_nshot.get_dataloaders(
        n_shot=args.n_shot, seed=args.seed, dataset_name=args.dataset
    )
    dataloader_S, _, val_S = dataset_source_full.get_dataloaders(seed=args.seed)
    
    num_classes_S = 10
    num_classes_R = 6 if args.dataset == "qiang" else 7
    
    print(len(dataloader_R), len(dataloader_S))

    model = DualHeadSwin3D(num_classes_S, num_classes_R).to(device)
    transform_S = TransformModule(
        lambda_residual=args.lambda_residual,
        disable_stn=args.disable_stn,
        disable_intensity=args.disable_intensity,
        disable_color=args.disable_color
    ).to(device)

    params = list(model.parameters()) + list(transform_S.parameters())

    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.02)

    loss_history = []
    epochs = 30
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 22], gamma=0.1)

    loss_history = {
        "total": [],
        "cls_S": [],
        "cls_R": [],
        "mmd": []
    }

    suffix = ""
    if args.disable_stn or args.disable_intensity or args.disable_color:
        parts = []
        if not args.disable_stn: parts.append("stn")
        if not args.disable_intensity: parts.append("intensity")
        if not args.disable_color: parts.append("color")
        suffix = "_ablation_" + "_".join(parts)

    LOG_PATH = f"experiment_log_{args.dataset}_lambda{args.lambda_mmd}_joint_{args.n_shot}shot_all_transforms_swin3d_{args.loss_type}{suffix}.csv"

    for epoch in range(epochs):
        loss = train_joint_alignment(
            model, transform_S, dataloader_S, dataloader_R, optimizer, scheduler,
            device, lambda_mmd=args.lambda_mmd, loss_type=args.loss_type, viz_dir=f"visuals_swin3d_residual_nshot_{args.dataset}", viz_interval=100, epoch=epoch
        )
        print(f"[Epoch {epoch}] Training loss = {loss}")

        if (epoch + 1) % 5 == 0:
            acc_S, acc_R = evaluate(model, val_S, val_R, device)
            print(f"Test Acc — Domain S: {acc_S:.2f}% | Domain R: {acc_R:.2f}%")

            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["seed", "n_shot", "lambda_residual", "lambda_mmd", "loss_type", "epoch", "acc_S", "acc_R"])
                writer.writerow([args.seed, args.n_shot, args.lambda_residual, args.lambda_mmd, args.loss_type, epoch + 1, acc_S, acc_R])

    model_name = f"saved_models/swin3d_joint_{args.dataset}_nshot{args.n_shot}_lambdares{args.lambda_residual}_lambdammd{args.lambda_mmd}_seed{args.seed}_epoch{epoch+1}_all_transforms_{args.loss_type}{suffix}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train joint n-shot Swin3D model with all transforms')
    parser.add_argument('--n_shot', type=int, default=3, help='Number of shots for few-shot learning')
    parser.add_argument('--lambda_residual', type=float, default=0.5, help='Lambda value for residual connection')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cuda_device', type=int, default=2, help='CUDA device ID')
    parser.add_argument('--lambda_mmd', type=float, default=0.2, help='Lambda value for alignment loss')
    parser.add_argument('--loss_type', type=str, default='mmd', choices=['mmd', 'coral'], help='Type of domain alignment loss')
    parser.add_argument('--disable_stn', action='store_true', help='Disable Spatial Transformer Network transformation')
    parser.add_argument('--disable_intensity', action='store_true', help='Disable DoG/Brightness/Contrast/Gamma transformations')
    parser.add_argument('--disable_color', action='store_true', help='Disable global color transformation')
    parser.add_argument('--dataset', type=str, default='noble', choices=['noble', 'qiang'], help='Target dataset name')

    args = parser.parse_args()
    main_exp(args)
