import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

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

def mmd_loss(x, y):
    """Compute unbiased MMD loss with multi-scale RBF kernels."""
    Bx, By = x.size(0), y.size(0)
    assert Bx == By, "MMD requires equal batch sizes; use random sampling if unequal."

    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * xy

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


# ---------- CORAL Loss ----------

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


# ---------- Training Loop ----------

def train_da(
    model, dataloader_S, dataloader_R, optimizer, scheduler, device,
    lambda_align=0.2, loss_type='mmd', epoch=0
):
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0

    for step, ((s_batch, s_label), (r_batch, r_label)) in enumerate(zip(dataloader_S, dataloader_R)):
        s_batch, r_batch = s_batch.to(device), r_batch.to(device)
        s_label, r_label = s_label.to(device), r_label.to(device)

        # Forward (NO transform module applied)
        model.eval()
        logits_s, feat_s = model(s_batch, domain='S', return_features=True)
        model.train()
        logits_r, feat_r = model(r_batch, domain='R', return_features=True)

        # Loss calculation
        loss_s = 0.5 * ce_loss(logits_s, s_label)
        loss_r = ce_loss(logits_r, r_label)
        loss_cls = loss_s + loss_r

        if lambda_align > 1e-6 and feat_s.shape == feat_r.shape:
            if loss_type == 'mmd':
                loss_align = lambda_align * mmd_loss(feat_s, feat_r)
            elif loss_type == 'coral':
                loss_align = lambda_align * coral_loss(feat_s, feat_r)
            else:
                loss_align = 0
            loss_total = loss_cls + loss_align
        else:
            loss_total = loss_cls

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            print(f"[WARNING] NaN/Inf detected at epoch {epoch}, step {step}")
            continue

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss_total.item()

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


# ---------- Main Execution ----------

import dataset_target_nshot
import dataset_source_full
from torch.optim import lr_scheduler
import config_Noble
import config_simulated_c10

config_Noble.BATCH_SIZE = 4
config_simulated_c10.BATCH_SIZE = 4
dataset_target_nshot.BATCH_SIZE = 4
dataset_source_full.BATCH_SIZE = 4

def main_exp(args):
    set_seed(args.seed)

    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda")

    dataloader_R, _, val_R = dataset_target_nshot.get_dataloaders(n_shot=args.n_shot, seed=args.seed)
    dataloader_S, _, val_S = dataset_source_full.get_dataloaders(seed=args.seed)
    num_classes_S, num_classes_R = 10, 7

    model = DualHeadSwin3D(num_classes_S, num_classes_R).to(device)
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.02)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 22], gamma=0.1)

    LOG_PATH = f"experiment_log_da_baseline_{args.loss_type}_{args.n_shot}shot_swin3d.csv"

    for epoch in range(30):
        loss = train_da(
            model, dataloader_S, dataloader_R, optimizer, scheduler, device,
            lambda_align=args.lambda_align, loss_type=args.loss_type, epoch=epoch
        )
        print(f"[Epoch {epoch}] Training loss = {loss}")

        if (epoch + 1) % 5 == 0:
            acc_S, acc_R = evaluate(model, val_S, val_R, device)
            print(f"Test Acc — Domain S: {acc_S:.2f}% | Domain R: {acc_R:.2f}%")

            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["seed", "n_shot", "lambda_align", "loss_type", "epoch", "acc_S", "acc_R"])
                writer.writerow([args.seed, args.n_shot, args.lambda_align, args.loss_type, epoch + 1, acc_S, acc_R])

    model_name = f"saved_models/swin3d_da_{args.loss_type}_nshot{args.n_shot}_lambda{args.lambda_align}_seed{args.seed}_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train standard DA baseline (MMD/CORAL) Swin3D model')
    parser.add_argument('--n_shot', type=int, default=3, help='Number of shots for few-shot learning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cuda_device', type=int, default=2, help='CUDA device ID')
    parser.add_argument('--lambda_align', type=float, default=0.2, help='Lambda value for feature alignment loss')
    parser.add_argument('--loss_type', type=str, default='mmd', choices=['mmd', 'coral'], help='Type of domain alignment loss')

    args = parser.parse_args()
    main_exp(args)
