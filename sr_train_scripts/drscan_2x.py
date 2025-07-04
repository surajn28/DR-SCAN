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
from torch.nn import MSELoss
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

        self.scale_factor = 2
        self.hr_size = 48
        self.lr_size = self.hr_size // self.scale_factor
        self.init_lr = 2e-4
        self.batch_size = 16
        self.num_epochs = 4000
        self.weight_decay = 0
        self.betas = (0.9, 0.99)
        self.grad_clip = 0.5

        self.lr_milestones = [400, 800, 1200, 1600, 2000]
        self.lr_gamma = 0.7

        self.num_workers = 4
        self.seed = 42
        self.patience = 300              # early stopping

        self.num_groups_shallow = 6
        self.num_groups_deep = 10
        self.num_blocks_per_group = 18
        self.num_feats = 128

        self.save_dir = f"DualBranch_x{self.scale_factor}"
        self.save_freq = 100
        self.val_freq = 1

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.perceptual_weight = 0
        self.resume_checkpoint = f"DualBranch_x2/checkpoints/epoch_2200.pth"
        # self.resume_checkpoint = ""


config = Config()

os.makedirs(config.save_dir, exist_ok=True)
for sd in ["checkpoints", "samples", "plots"]:
    os.makedirs(os.path.join(config.save_dir, sd), exist_ok=True)


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_psnr = state["best_psnr"]
        logger.info(
            f"Resuming training at epoch {start_epoch} with best PSNR {best_psnr:.2f}"
        )
        return start_epoch, best_psnr
    logger.info("No checkpoint found, starting from scratch.")
    return 1, 0.0


# Model building functions

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        return x * attn


class RCASABlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_feats, n_feats, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            n_feats, n_feats, kernel_size, padding=kernel_size // 2)
        self.ca = ChannelAttention(n_feats, reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        out = self.sa(out)
        return identity + out


class ResidualGroup(nn.Module):
    def __init__(self, n_feats, n_blocks, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            *[RCASABlock(n_feats, 3, reduction) for _ in range(n_blocks)]
        )
        self.conv_last = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def forward(self, x):
        res = x
        out = self.conv_last(self.body(x))
        return res + out


class DualBranch(nn.Module):
    def __init__(
        self,
        scale_factor=3,
        in_channels=3,
        n_feats=64,
        num_groups_shallow=4,
        num_groups_deep=8,
        num_blocks_per_group=12,
        reduction=16,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # shallow branch
        self.shallow_head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.shallow_groups = nn.ModuleList(
            [
                ResidualGroup(n_feats, num_blocks_per_group, reduction)
                for _ in range(num_groups_shallow)
            ]
        )
        self.shallow_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # deep branch
        self.deep_head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.deep_groups = nn.ModuleList(
            [
                ResidualGroup(n_feats, num_blocks_per_group, reduction)
                for _ in range(num_groups_deep)
            ]
        )
        self.deep_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

        # upsample head
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale_factor**2, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(n_feats, in_channels, 3, padding=1),
        )

    def forward(self, x):
        # shallow
        s = self.shallow_head(x)
        for g in self.shallow_groups:
            s = g(s)
        s = self.shallow_tail(s)

        # deep
        d = self.deep_head(x)
        for g in self.deep_groups:
            d = g(d)
        d = self.deep_tail(d)

        # fuse & upsample
        out = self.fuse(torch.cat([s, d], 1))
        out = self.upsample(out)

        # global skip (0.2)
        skip = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )
        return out + 0.2 * skip

# Dataset - Div2k and Flickr2k


class CombinedSRDataset(Dataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.cfg = config
        self.is_train = split == "train"
        self.hr_files, self.lr_files = self._collect_files(split)

        self.hr_crop = config.hr_size
        self.lr_crop = self.hr_crop // config.scale_factor
        self.to_tensor = transforms.ToTensor()

    def _collect_files(self, split):
        if split == "train":
            div2k_hr = sorted(
                glob.glob(os.path.join(self.cfg.div2k_path,
                          "DIV2K_train_HR", "*.png"))
            )
            div2k_lr = sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.div2k_path,
                        f"DIV2K_train_LR_bicubic/X{self.cfg.scale_factor}",
                        "*.png",
                    )
                )
            )
            flickr_hr = sorted(
                glob.glob(os.path.join(
                    self.cfg.flickr2k_path, "Flickr2K_HR", "*.png"))
            )
            flickr_lr = sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.flickr2k_path,
                        f"Flickr2K_LR_bicubic/X{self.cfg.scale_factor}",
                        "*.png",
                    )
                )
            )
            return div2k_hr + flickr_hr, div2k_lr + flickr_lr

        hr = sorted(
            glob.glob(os.path.join(self.cfg.div2k_path,
                      "DIV2K_valid_HR", "*.png"))
        )
        lr = sorted(
            glob.glob(
                os.path.join(
                    self.cfg.div2k_path,
                    f"DIV2K_valid_LR_bicubic/X{self.cfg.scale_factor}",
                    "*.png",
                )
            )
        )
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

        # choose top-left on the LR grid
        x_lr = random.randint(0, w_hr // s - self.lr_crop)
        y_lr = random.randint(0, h_hr // s - self.lr_crop)
        x_hr, y_hr = x_lr * s, y_lr * s

        hr_patch = hr_img.crop(
            (x_hr, y_hr, x_hr + self.hr_crop, y_hr + self.hr_crop)
        )
        lr_patch = lr_img.crop(
            (x_lr, y_lr, x_lr + self.lr_crop, y_lr + self.lr_crop)
        )
        return hr_patch, lr_patch

    def _center_crop(self, hr_img, lr_img):
        s = self.cfg.scale_factor
        w_hr, h_hr = hr_img.size

        # centre in LR space → map to HR
        x_lr = (w_hr // s - self.lr_crop) // 2
        y_lr = (h_hr // s - self.lr_crop) // 2
        x_hr, y_hr = x_lr * s, y_lr * s

        hr_patch = hr_img.crop(
            (x_hr, y_hr, x_hr + self.hr_crop, y_hr + self.hr_crop)
        )
        lr_patch = lr_img.crop(
            (x_lr, y_lr, x_lr + self.lr_crop, y_lr + self.lr_crop)
        )
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

# Metrics


def calculate_psnr(sr, hr):
    sr = np.clip(sr.detach().cpu().numpy(), 0.0, 1.0)
    hr = np.clip(hr.detach().cpu().numpy(), 0.0, 1.0)
    mse = np.mean((sr - hr) ** 2)
    if mse < 1e-10:
        return 100
    return -10 * math.log10(mse)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        for p in vgg.parameters():
            p.requires_grad = False
        self.slice1, self.slice2 = vgg[:2], vgg[2:7]
        self.slice3, self.slice4 = vgg[7:12], vgg[12:21]
        self.slice5 = vgg[21:30]
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, sr, hr):
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        loss = 0
        for s_slice, h_slice in zip(
            (self.slice1, self.slice2, self.slice3, self.slice4, self.slice5),
            (self.slice1, self.slice2, self.slice3, self.slice4, self.slice5),
        ):
            sr, hr = s_slice(sr), h_slice(hr)
            loss += F.l1_loss(sr, hr)
        return loss

# Validation


@torch.no_grad()
def validate(model, loader, pix_loss, perc_loss, cfg):
    model.eval()
    tot_loss, tot_psnr, n = 0.0, 0.0, 0
    for lr, hr in loader:
        lr, hr = lr.to(cfg.device), hr.to(cfg.device)
        sr = model(lr)
        loss = pix_loss(sr, hr) + cfg.perceptual_weight * perc_loss(sr, hr)
        tot_loss += loss.item()
        tot_psnr += calculate_psnr(sr, hr)
        n += 1
    model.train()
    return tot_loss / n, tot_psnr / n


def save_ckpt(epoch, model, opt, sch, best_psnr, cfg, is_best=False):
    ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sch.state_dict() if sch else None,
        "best_psnr": best_psnr,
    }
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))
    if epoch % cfg.save_freq == 0:
        torch.save(state, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

# Training loop


def train(cfg):
    set_random_seeds(cfg.seed)

    train_ds = CombinedSRDataset(cfg, "train")
    val_ds = CombinedSRDataset(cfg, "valid")
    logger.info(f"Train set: {len(train_ds)} | Val set: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    model = DualBranch(
        scale_factor=cfg.scale_factor,
        in_channels=3,
        n_feats=cfg.num_feats,
        num_groups_shallow=cfg.num_groups_shallow,
        num_groups_deep=cfg.num_groups_deep,
        num_blocks_per_group=cfg.num_blocks_per_group,
    ).to(cfg.device)
    logger.info(
        f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # pix_loss = MSELoss().to(cfg.device)
    pix_loss = nn.L1Loss().to(cfg.device)  # L1-Loss
    perc_loss = VGGPerceptualLoss().to(cfg.device)

    opt = optim.Adam(
        model.parameters(),
        lr=cfg.init_lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    sch = MultiStepLR(opt, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)

    start_epoch, best_psnr = 1, 0.0
    if os.path.isfile(cfg.resume_checkpoint):
        start_epoch, best_psnr = load_checkpoint(
            cfg.resume_checkpoint, model, opt, sch, cfg.device
        )

    patience_ctr, best_epoch = 0, start_epoch
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        ep_loss, ep_psnr, steps = 0.0, 0.0, 0

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
            steps += 1

        sch.step()

        tr_loss, tr_psnr = ep_loss / steps, ep_psnr / steps

        if epoch % cfg.val_freq == 0:
            val_loss, val_psnr = validate(
                model, val_loader, pix_loss, perc_loss, cfg)
            logger.info(
                f"Ep {epoch:4d}/{cfg.num_epochs} | "
                f"Train L={tr_loss:.4f} P={tr_psnr:.2f} | "
                f"Val L={val_loss:.4f} P={val_psnr:.2f} | "
                f"LR={opt.param_groups[0]['lr']:.1e}"
            )

            improved = val_psnr > best_psnr
            if improved:
                best_psnr, best_epoch, patience_ctr = val_psnr, epoch, 0
            else:
                patience_ctr += 1

            save_ckpt(epoch, model, opt, sch, best_psnr, cfg, improved)
        else:
            logger.info(
                f"Ep {epoch:4d}/{cfg.num_epochs} | "
                f"Train L={tr_loss:.4f} P={tr_psnr:.2f} | "
                f"LR={opt.param_groups[0]['lr']:.1e}"
            )

        if patience_ctr >= cfg.patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Done. Best PSNR={best_psnr:.2f} at epoch {best_epoch}")
    return model


if __name__ == "__main__":
    set_random_seeds(config.seed)
    train(config)
