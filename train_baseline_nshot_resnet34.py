import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import csv
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend suitable for saving to files
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
import math
from functools import partial

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


import resnet
SAMPLE_DURATION = 28 #32 for simulated data

class resnet34_3d(nn.Module):
    def __init__(self, model, hidden_size, num_classes):
        super(resnet34_3d, self).__init__()
        self.model = model
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

def load_resnet_video_pretrained(model_path):
    model = resnet.resnet34(num_classes=400, shortcut_type='A',
                            sample_size=128, sample_duration=SAMPLE_DURATION,
                            last_fc=False)

    if model_path != None and model_path != "":
        weights = torch.load(model_path, map_location='cpu')
        if "state_dict" in weights:
            model.load_state_dict(weights["state_dict"])
        else:
            model.load_state_dict(weights)
        print(f"loaded weight from {model_path}")
    return model


# ---------- Differentiable Transform Modules ----------

def color_shift(x, color_matrix, bias):
    B, C, D, H, W = x.shape
    mean = x.mean(dim=[2, 3, 4], keepdim=True)
    x_centered = x - mean
    x_trans = torch.einsum('bcdhw,bij->bdhwi', x_centered, color_matrix)
    x_trans = x_trans.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
    x_trans = x_trans + bias.view(B, C, 1, 1, 1) + mean
    return x_trans.clamp(0, 1)


class DeterministicAvgPool3d(nn.Module):
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
        patches = x.unfold(2, kD, sD).unfold(3, kH, sH).unfold(4, kW, sW)
        return patches.mean(dim=(-3, -2, -1))


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
            DeterministicAvgPool3d(2, stride=2),
            nn.Conv3d(8, 10, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1,0,0,0, 0,1,0,0, 0,0,1,0], dtype=torch.float)
        )

        self.fc_color = nn.Sequential(
            nn.Linear(10 * 4 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)
        )
        self.fc_color[2].weight.data.zero_()
        self.fc_color[2].bias.data.copy_(
            torch.tensor([1,0,0, 0,1,0, 0,0,1, 0,0,0], dtype=torch.float)
        )

    @staticmethod
    def _deterministic_avg_pool3d(x, kernel_size, stride):
        kD, kH, kW = kernel_size
        sD, sH, sW = stride
        patches = x.unfold(2, kD, sD).unfold(3, kH, sH).unfold(4, kW, sW)
        return patches.mean(dim=(-3, -2, -1))

    @staticmethod
    def _deterministic_adaptive_avg_pool3d(x, output_size):
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        oD, oH, oW = output_size
        kD, kH, kW = D // oD, H // oH, W // oW
        return SpatialTransformer._deterministic_avg_pool3d(x, (kD, kH, kW), (kD, kH, kW))

    @staticmethod
    def _deterministic_grid_sample_3d(input, grid):
        B, C, D, H, W = input.shape
        _, oD, oH, oW, _ = grid.shape

        grid_x = ((grid[..., 0] + 1) / 2) * (W - 1)
        grid_y = ((grid[..., 1] + 1) / 2) * (H - 1)
        grid_z = ((grid[..., 2] + 1) / 2) * (D - 1)

        x0 = grid_x.floor().long().clamp(0, W - 1)
        x1 = (x0 + 1).clamp(0, W - 1)
        y0 = grid_y.floor().long().clamp(0, H - 1)
        y1 = (y0 + 1).clamp(0, H - 1)
        z0 = grid_z.floor().long().clamp(0, D - 1)
        z1 = (z0 + 1).clamp(0, D - 1)

        wx = (grid_x - grid_x.floor()).unsqueeze(1)
        wy = (grid_y - grid_y.floor()).unsqueeze(1)
        wz = (grid_z - grid_z.floor()).unsqueeze(1)

        def gather_nd(t, z, y, x):
            t_flat = t.view(B, C, -1)
            idx = z * H * W + y * W + x
            idx = idx.view(B, 1, -1).expand(B, C, -1)
            gathered = torch.gather(t_flat, 2, idx)
            return gathered.view(B, C, oD, oH, oW)

        c000 = gather_nd(input, z0, y0, x0)
        c001 = gather_nd(input, z0, y0, x1)
        c010 = gather_nd(input, z0, y1, x0)
        c011 = gather_nd(input, z0, y1, x1)
        c100 = gather_nd(input, z1, y0, x0)
        c101 = gather_nd(input, z1, y0, x1)
        c110 = gather_nd(input, z1, y1, x0)
        c111 = gather_nd(input, z1, y1, x1)

        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        return c0 * (1 - wz) + c1 * wz

    def forward(self, x):
        loc_feat = self.localization(x)
        xs = self._deterministic_adaptive_avg_pool3d(loc_feat, (4, 4, 4))
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 3, 4)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_stn = self._deterministic_grid_sample_3d(x, grid)

        p_color = self.fc_color(xs).view(-1, 4, 3)
        color_matrix = p_color[:, :3, :]
        bias = p_color[:, 3, :]
        return color_shift(x_stn, color_matrix, bias)


class DoGFilter3D(nn.Module):
    def __init__(self, sigma1=1.0, sigma2=2.0, channels=3):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(sigma1))
        self.sigma2 = nn.Parameter(torch.tensor(sigma2))
        self.channels = channels

    def gaussian_kernel(self, sigma):
        size = int(2 * (3 * sigma) + 1)
        if size % 2 == 0:
            size += 1
        size = min(size, 15)
        x = torch.arange(size).float() - size // 2
        x = x.to(sigma.device)
        g = torch.exp(-x ** 2 / (2 * sigma ** 2 + 1e-8))
        return g / (g.sum() + 1e-8)

    def forward(self, x):
        sigma1 = self.sigma1.clamp(0.5, 5.0)
        sigma2 = self.sigma2.clamp(0.5, 5.0)
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
        return ((x - mean) * self.contrast + mean + self.brightness).clamp(0, 1)


class GammaAdjust(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x, self.gamma.clamp(0.5, 2.0))


class GlobalColorTransform(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.color_matrix = nn.Parameter(torch.eye(num_channels) + 0.01 * torch.randn(num_channels, num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        B, C, D, H, W = x.shape
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        x_centered = x - mean
        x_trans = torch.tensordot(x_centered, self.color_matrix, dims=([1],[1])).permute(0,4,1,2,3)
        return (x_trans + self.bias.view(1, C, 1, 1, 1) + mean).clamp(0, 1)


class TransformModule(nn.Module):
    def __init__(self, lambda_residual=0.5):
        super().__init__()
        self.stn = SpatialTransformer()
        self.dog = DoGFilter3D()
        self.bc = BrightnessContrastAdjust()
        self.gamma = GammaAdjust()
        self.color = GlobalColorTransform()

        self.register_buffer('lambda_stn', torch.tensor(lambda_residual))
        self.register_buffer('lambda_dog', torch.tensor(lambda_residual))
        self.register_buffer('lambda_bc', torch.tensor(lambda_residual))
        self.register_buffer('lambda_gamma', torch.tensor(lambda_residual))
        self.register_buffer('lambda_color', torch.tensor(lambda_residual))

    def forward(self, x):
        # Structurally same as swin3d baseline: defines transforms but only applies STN in baseline config if needed
        x_orig = x
        x = self.lambda_stn.clamp(0, 1) * x + (1 - self.lambda_stn.clamp(0, 1)) * self.stn(x)
        return x


# ---------- Shared Backbone + Two Heads (3D ResNet-34) ----------

class DualHeadResNet34(nn.Module):
    def __init__(self, num_classes_S, num_classes_R, pretrained_path='resnet-34-kinetics-cpu.pth'):
        super().__init__()
        video_pretrained_model = load_resnet_video_pretrained(pretrained_path)
        backbone = resnet34_3d(video_pretrained_model, 512, num_classes_R)
        
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_S = nn.Linear(in_feat, num_classes_S)
        self.head_R = nn.Linear(in_feat, num_classes_R)

    def forward(self, x, domain='S', return_features=False):
        feats = self.backbone(x)
        logits = self.head_S(feats) if domain == 'S' else self.head_R(feats)
        return (logits, feats) if return_features else logits

DualHeadResNet3D = DualHeadResNet34


# ---------- MMD Loss ----------

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

def train_real(
    model, dataloader_R, optimizer, scheduler, device,
    lambda_mmd=0.5, viz_dir="visuals", viz_interval=200, epoch=0
):
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0

    for step, (r_batch, r_label) in enumerate(dataloader_R):
        r_batch = r_batch.to(device)
        r_label = r_label.to(device)

        # Transform and forward
        model.train()
        logits_r, feat_r = model(r_batch, domain='R', return_features=True)

        # Loss
        loss_total = ce_loss(logits_r, r_label)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        total_loss += loss_total.item()

    scheduler.step()
    return total_loss / len(dataloader_R)


# ---------- Evaluation Function ----------

@torch.no_grad()
def evaluate(model, dataloader_R_val, device):
    model.eval()
    correct_R = total_R = 0

    for x, y in dataloader_R_val:
        x, y = x.to(device), y.to(device)
        preds = model(x, domain='R').argmax(dim=1)
        correct_R += (preds == y).sum().item()
        total_R += y.size(0)

    acc_R = 100 * correct_R / total_R if total_R > 0 else 0
    return acc_R


# ---------- Main Execution ----------

import dataset_target_nshot
from torch.optim import lr_scheduler

dataset_target_nshot.BATCH_SIZE = 4

def main_exp(args):
    set_seed(args.seed)

    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda")

    # Import target config dynamically and override batch size
    if args.dataset == "qiang":
        import config_Qiang as target_config
    else:
        import config_Noble as target_config
    target_config.BATCH_SIZE = 4

    dataloader_R, _, val_R = dataset_target_nshot.get_dataloaders(
        n_shot=args.n_shot, seed=args.seed, dataset_name=args.dataset
    )
    num_classes_R = 6 if args.dataset == "qiang" else 7

    model = DualHeadResNet34(0, num_classes_R, pretrained_path=args.pretrained_path).to(device)
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.02)
    
    loss_history = []
    epochs = 30
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 22], gamma=0.1)

    loss_history = {
        "total": [],
        "cls_S": [],
        "cls_R": [],
        "mmd": []
    }

    LOG_PATH = f"experiment_log_{args.dataset}_lambda{args.lambda_mmd}_baseline_{args.n_shot}shot_resnet34.csv"

    for epoch in range(epochs):
        loss = train_real(model, dataloader_R, optimizer, scheduler, device,
                          lambda_mmd=args.lambda_mmd, viz_interval=100, epoch=epoch)
        print(f"[Epoch {epoch}] Training loss = {loss}")

        if (epoch + 1) % 5 == 0:
            acc_R = evaluate(model, val_R, device)
            print(f"Test Acc — Domain R: {acc_R:.2f}%")

            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["seed", "n_shot", "lambda_residual", "lambda_mmd", "epoch", "acc_R"])
                writer.writerow([args.seed, args.n_shot, args.lambda_residual, args.lambda_mmd, epoch + 1, acc_R])

    model_name = f"saved_models/resnet34_baseline_{args.dataset}_nshot{args.n_shot}_lambdares{args.lambda_residual}_lambdammd{args.lambda_mmd}_seed{args.seed}_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline n-shot ResNet-34 model')
    parser.add_argument('--n_shot', type=int, default=3, help='Number of shots for few-shot learning')
    parser.add_argument('--lambda_residual', type=float, default=0.5, help='Lambda value for residual connection')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cuda_device', type=int, default=7, help='CUDA device ID')
    parser.add_argument('--lambda_mmd', type=float, default=0.2, help='Lambda value for MMD loss')
    parser.add_argument('--pretrained_path', type=str, default='resnet-34-kinetics-cpu.pth', help='Path to resnet-34 pretrained weights')
    parser.add_argument('--dataset', type=str, default='noble', choices=['noble', 'qiang'], help='Target dataset name')

    args = parser.parse_args()
    main_exp(args)
