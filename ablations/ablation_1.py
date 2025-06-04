
"""
Single‑Branch (Ablation-1)
 - One residual branch (16 groups) instead of dual shallow/deep.
  - No Channel/Spatial attention.
Everything else – dataset, training loop, loss, schedule – is identical
to the baseline Dual‑Branch script.
"""
import os
import math
import random
import glob
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import MultiStepLR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.div2k_path = "/mimer/NOBACKUP/groups/geodl/DIV2K/"
        self.flickr2k_path = "/mimer/NOBACKUP/groups/geodl/Flickr2K/"

        # Scale & patch size
        self.scale_factor = 4
        self.hr_size = 192
        self.lr_size = self.hr_size // self.scale_factor

        # Training hyper‑parameters
        self.init_lr = 2e-4
        self.batch_size = 16
        self.num_epochs = 2000
        self.weight_decay = 0.0
        self.betas = (0.9, 0.99)
        self.grad_clip = 0.5
        self.lr_milestones = [400, 800, 1200, 1600]
        self.lr_gamma = 0.7
        self.num_workers = 4
        self.seed = 42
        self.patience = 300

        self.num_groups = 16          # 6 + 10 from baseline
        self.num_blocks_per_group = 12
        self.num_feats = 128

        self.save_dir = f"SingleBranch_RCAN_x{self.scale_factor}"
        self.save_freq = 100
        self.val_freq = 1
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.perceptual_weight = 0.05
        self.resume_checkpoint = ""


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
for sd in ["checkpoints", "samples", "plots"]:
    os.makedirs(os.path.join(cfg.save_dir, sd), exist_ok=True)


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

#  Plain Residial block & Residual Group (no attention)


class PlainResBlock(nn.Module):
    """Residual Channel‑Attention Block WITHOUT attention (just conv‑ReLU‑conv)."""

    def __init__(self, n_feats, kernel_size=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        )

    def forward(self, x):
        return x + self.body(x)


class PlainResGroup(nn.Module):
    def __init__(self, n_feats, n_blocks):
        super().__init__()
        modules = [PlainResBlock(n_feats) for _ in range(n_blocks)]
        modules.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        return x + res

# Single‑Branch SR network


class SingleBranch(nn.Module):
    def __init__(self,
                 scale_factor=4,
                 in_channels=3,
                 n_feats=128,
                 num_groups=16,
                 num_blocks_per_group=12):
        super().__init__()
        self.scale = scale_factor

        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.groups = nn.ModuleList(
            [PlainResGroup(n_feats, num_blocks_per_group)
             for _ in range(num_groups)]
        )
        self.tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # Upsampler
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale_factor**2, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(n_feats, in_channels, 3, padding=1),
        )

    def forward(self, x):
        out = self.head(x)
        for g in self.groups:
            out = g(out)
        out = self.tail(out)
        out = self.upsample(out)

        skip = F.interpolate(x, scale_factor=self.scale,
                             mode="bicubic", align_corners=False)
        return out + 0.2 * skip  # keep same global skip as baseline

# Dataset


class CombinedSRDataset(Dataset):
    def __init__(self, cfg, split="train"):
        super().__init__()
        self.cfg = cfg
        self.is_train = split == "train"
        self.hr_files, self.lr_files = self._collect_files(split)

        self.hr_crop = cfg.hr_size
        self.lr_crop = self.hr_crop // cfg.scale_factor
        self.to_tensor = transforms.ToTensor()

    def _collect_files(self, split):
        if split == "train":
            div2k_hr = sorted(glob.glob(os.path.join(
                cfg.div2k_path, "DIV2K_train_HR", "*.png")))
            div2k_lr = sorted(glob.glob(os.path.join(
                cfg.div2k_path, f"DIV2K_train_LR_bicubic/X{cfg.scale_factor}", "*.png")))
            flickr_hr = sorted(glob.glob(os.path.join(
                cfg.flickr2k_path, "Flickr2K_HR", "*.png")))
            flickr_lr = sorted(glob.glob(os.path.join(
                cfg.flickr2k_path, f"Flickr2K_LR_bicubic/X{cfg.scale_factor}", "*.png")))
            return div2k_hr + flickr_hr, div2k_lr + flickr_lr

        hr = sorted(glob.glob(os.path.join(
            cfg.div2k_path, "DIV2K_valid_HR", "*.png")))
        lr = sorted(glob.glob(os.path.join(
            cfg.div2k_path, f"DIV2K_valid_LR_bicubic/X{cfg.scale_factor}", "*.png")))
        return hr, lr

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")

        if self.is_train:
            hr, lr = self._random_crop(hr, lr)
            hr, lr = self._augment(hr, lr)
        else:
            hr, lr = self._center_crop(hr, lr)

        return self.to_tensor(lr), self.to_tensor(hr)

    def _random_crop(self, hr_img, lr_img):
        s = self.cfg.scale_factor
        w_hr, h_hr = hr_img.size
        x_lr = random.randint(0, w_hr // s - self.lr_crop)
        y_lr = random.randint(0, h_hr // s - self.lr_crop)
        x_hr, y_hr = x_lr * s, y_lr * s
        hr_patch = hr_img.crop(
            (x_hr, y_hr, x_hr + self.hr_crop, y_hr + self.hr_crop))
        lr_patch = lr_img.crop(
            (x_lr, y_lr, x_lr + self.lr_crop, y_lr + self.lr_crop))
        return hr_patch, lr_patch

    def _center_crop(self, hr_img, lr_img):
        s = self.cfg.scale_factor
        w_hr, h_hr = hr_img.size
        x_lr = (w_hr // s - self.lr_crop) // 2
        y_lr = (h_hr // s - self.lr_crop) // 2
        x_hr, y_hr = x_lr * s, y_lr * s
        hr_patch = hr_img.crop(
            (x_hr, y_hr, x_hr + self.hr_crop, y_hr + self.hr_crop))
        lr_patch = lr_img.crop(
            (x_lr, y_lr, x_lr + self.lr_crop, y_lr + self.lr_crop))
        return hr_patch, lr_patch

    @staticmethod
    def _augment(hr, lr):
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
        return hr, lr


#  Metrics and Losses


def calculate_psnr(sr, hr):
    sr = np.clip(sr.detach().cpu().numpy(), 0.0, 1.0)
    hr = np.clip(hr.detach().cpu().numpy(), 0.0, 1.0)
    mse = np.mean((sr - hr) ** 2)
    return 100 if mse < 1e-10 else -10 * math.log10(mse)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        for p in vgg.parameters():
            p.requires_grad = False
        self.slices = nn.ModuleList(
            [vgg[:2], vgg[2:7], vgg[7:12], vgg[12:21], vgg[21:30]])
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr, hr = (sr - self.mean)/self.std, (hr - self.mean)/self.std
        loss = 0.0
        for sl in self.slices:
            sr, hr = sl(sr), sl(hr)
            loss += F.l1_loss(sr, hr)
        return loss


@torch.no_grad()
def validate(model, loader, pix_loss, perc_loss):
    model.eval()
    tot_loss = tot_psnr = 0.0
    for lr, hr in loader:
        lr, hr = lr.to(cfg.device), hr.to(cfg.device)
        sr = model(lr)
        loss = pix_loss(sr, hr) + cfg.perceptual_weight * perc_loss(sr, hr)
        tot_loss += loss.item()
        tot_psnr += calculate_psnr(sr, hr)
    model.train()
    n = len(loader)
    return tot_loss / n, tot_psnr / n


def save_ckpt(epoch, model, opt, sch, best_psnr, is_best=False):
    ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sch.state_dict(),
        "best_psnr": best_psnr,
    }
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))
    if epoch % cfg.save_freq == 0:
        torch.save(state, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

#  Training loop


def train():
    set_random_seeds(cfg.seed)

    train_ds = CombinedSRDataset(cfg, "train")
    val_ds = CombinedSRDataset(cfg, "valid")
    logger.info(f"Train set: {len(train_ds)} | Val set: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = SingleBranch(scale_factor=cfg.scale_factor,
                         n_feats=cfg.num_feats,
                         num_groups=cfg.num_groups,
                         num_blocks_per_group=cfg.num_blocks_per_group).to(cfg.device)
    logger.info(
        f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    pix_loss = nn.L1Loss().to(cfg.device)
    perc_loss = VGGPerceptualLoss().to(cfg.device)

    opt = optim.Adam(model.parameters(), lr=cfg.init_lr,
                     betas=cfg.betas, weight_decay=cfg.weight_decay)
    sch = MultiStepLR(opt, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)

    start_epoch, best_psnr, patience_ctr = 1, 0.0, 0

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        ep_loss = ep_psnr = 0.0
        for lr, hr in train_loader:
            lr, hr = lr.to(cfg.device), hr.to(cfg.device)
            opt.zero_grad()
            sr = model(lr)
            loss = pix_loss(sr, hr) + cfg.perceptual_weight * perc_loss(sr, hr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
            ep_psnr += calculate_psnr(sr, hr)
        sch.step()

        tr_loss, tr_psnr = ep_loss / \
            len(train_loader), ep_psnr / len(train_loader)

        if epoch % cfg.val_freq == 0:
            val_loss, val_psnr = validate(
                model, val_loader, pix_loss, perc_loss)
            logger.info(f"Ep {epoch:4d}/{cfg.num_epochs} | "
                        f"Train L={tr_loss:.4f} P={tr_psnr:.2f} | "
                        f"Val L={val_loss:.4f} P={val_psnr:.2f} | "
                        f"LR={opt.param_groups[0]['lr']:.1e}")
            improved = val_psnr > best_psnr
            if improved:
                best_psnr, patience_ctr = val_psnr, 0
            else:
                patience_ctr += 1
            save_ckpt(epoch, model, opt, sch, best_psnr, improved)
        else:
            logger.info(f"Ep {epoch:4d}/{cfg.num_epochs} | "
                        f"Train L={tr_loss:.4f} P={tr_psnr:.2f} | "
                        f"LR={opt.param_groups[0]['lr']:.1e}")

        if patience_ctr >= cfg.patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Done. Best PSNR={best_psnr:.2f}")


if __name__ == "__main__":
    train()
