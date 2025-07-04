# ui_evaluate_sr.py

import streamlit as st
import io
import os
import numpy as np
from PIL import Image
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import lpips

st.set_page_config(
    layout="wide", page_title="DR-SCAN Interactive Branch Weighting")

# Model imports
try:
    # Must point to the same dr_scan.py files
    from drscan_2x import DualBranch, config
    st.success("Model code imported successfully!")
except ImportError as e:
    st.error(f"Could not import DualBranchRCAN: {e}")
    st.stop()

# Weighted‐fusion forward pass


def forward_weighted_fusion(
    self, x,
    *, shallow_weight: float = 1.0,
    deep_weight:    float = 1.0,
    fusion_scale:  float = 1.0
):
    with torch.no_grad():
        # 1) shallow branch
        s = self.shallow_head(x)
        for g in self.shallow_groups:
            s = g(s)
        s = self.shallow_tail(s)

        # 2) deep branch
        d = self.deep_head(x)
        for g in self.deep_groups:
            d = g(d)
        d = self.deep_tail(d)

        # 3) fusion core
        fused = torch.cat([s, d], dim=1)
        if hasattr(self, "fuse"):
            # new‐style fuse block in drscan_4x.py
            fused = self.fuse(fused)
        else:
            # legacy option if you ever load a checkpoint
            fused = self.fusion_conv(fused)
            fused = self.fusion_relu(fused)
            fused = self.fusion_tail(fused)

        # 4) branch re-weighting
        #    fusion_scale globally amplifies/attenuates your S/D offsets
        fused += (shallow_weight - 1.0) * fusion_scale * s
        fused += (deep_weight - 1.0) * fusion_scale * d

        # 5) upsample + fixed skip=0.2
        out = self.upsample(fused)
        skip = F.interpolate(x, scale_factor=self.scale_factor,
                             mode="bicubic", align_corners=False)
        return torch.clamp(out + 0.2*skip, 0, 1)


DualBranch.forward_weighted_fusion = forward_weighted_fusion


@dataclass
class BranchWeights:
    shallow: float
    deep: float


# LPIPS network

lpips_net = lpips.LPIPS(net='alex').eval()


def calc_lpips(sr_np: np.ndarray, hr_np: np.ndarray) -> float:
    sr_t = torch.from_numpy(sr_np.transpose(2, 0, 1)).unsqueeze(0).float()
    hr_t = torch.from_numpy(hr_np.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        return lpips_net(sr_t, hr_t).item()

# Test Time Augmentation if needed


def augment_image(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    elif mode == 1:
        return torch.flip(x, dims=[3])
    elif mode == 2:
        return torch.flip(x, dims=[2])
    elif mode == 3:
        return torch.flip(x, dims=[2, 3])
    elif mode == 4:
        return torch.rot90(x, k=1, dims=[2, 3])
    elif mode == 5:
        return torch.flip(torch.rot90(x, 1, [2, 3]), dims=[3])
    elif mode == 6:
        return torch.flip(torch.rot90(x, 1, [2, 3]), dims=[2])
    elif mode == 7:
        return torch.rot90(x, k=3, dims=[2, 3])
    else:
        return x


def invert_augment_image(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    elif mode == 1:
        return torch.flip(x, dims=[3])
    elif mode == 2:
        return torch.flip(x, dims=[2])
    elif mode == 3:
        return torch.flip(x, dims=[2, 3])
    elif mode == 4:
        return torch.rot90(x, k=3, dims=[2, 3])
    elif mode == 5:
        tmp = torch.flip(x, dims=[3])
        return torch.rot90(tmp, 3, [2, 3])
    elif mode == 6:
        tmp = torch.flip(x, dims=[2])
        return torch.rot90(tmp, 3, [2, 3])
    elif mode == 7:
        return torch.rot90(x, k=1, dims=[2, 3])
    else:
        return x


class DualBranchProcessor:
    def __init__(self, model: DualBranch, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.to_tensor = transforms.ToTensor()
        self.scale = model.scale_factor

    def preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        return self.to_tensor(pil_img).unsqueeze(0).to(self.device)

    def postprocess(self, sr_tensor: torch.Tensor) -> np.ndarray:
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
        return sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def baseline_forward(self, lr: torch.Tensor, use_tta=False) -> torch.Tensor:
        with torch.no_grad():
            if not use_tta:
                return torch.clamp(self.model(lr), 0, 1)
            outs = []
            for m in range(8):
                aug = augment_image(lr, m)
                sr = torch.clamp(self.model(aug), 0, 1)
                outs.append(invert_augment_image(sr, m))
            return torch.mean(torch.stack(outs, 0), 0, keepdim=True)

    def weighted_forward(self, lr: torch.Tensor,
                         w: BranchWeights, fusion_scale: float,
                         use_tta=False) -> torch.Tensor:
        with torch.no_grad():
            if not use_tta:
                return torch.clamp(
                    self.model.forward_weighted_fusion(
                        lr,
                        shallow_weight=w.shallow,
                        deep_weight=w.deep,
                        fusion_scale=fusion_scale
                    ), 0, 1)
            outs = []
            for m in range(8):
                aug = augment_image(lr, m)
                sr = self.model.forward_weighted_fusion(
                    aug,
                    shallow_weight=w.shallow,
                    deep_weight=w.deep,
                    fusion_scale=fusion_scale
                )
                outs.append(invert_augment_image(torch.clamp(sr, 0, 1), m))
            return torch.mean(torch.stack(outs, 0), 0, keepdim=True)

    def local_search_lpips(self, lr: torch.Tensor, hr: torch.Tensor,
                           fusion_scale: float,
                           steps: int = 5, step_size: float = 0.1) -> BranchWeights:
        hr_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
        cur = BranchWeights(1.0, 1.0)
        best_np = self.postprocess(
            self.weighted_forward(lr, cur, fusion_scale, False))
        best_lp = calc_lpips(best_np, hr_np)
        offsets = [(step_size, 0), (-step_size, 0),
                   (0, step_size), (0, -step_size),
                   (step_size, step_size), (step_size, -step_size),
                   (-step_size, step_size), (-step_size, -step_size)]
        it = 0
        improved = True
        while improved and it < steps:
            improved = False
            it += 1
            for dx, dy in offsets:
                cand = BranchWeights(
                    float(np.clip(cur.shallow+dx, 0.5, 2.0)),
                    float(np.clip(cur.deep + dy, 0.5, 2.0))
                )
                sr_np = self.postprocess(
                    self.weighted_forward(lr, cand, fusion_scale, False))
                lp = calc_lpips(sr_np, hr_np)
                if lp < best_lp:
                    best_lp = lp
                    cur = cand
                    improved = True
                    break
        return cur


# Streamlit layout

st.title("DR-SCAN: Interactive Branch Weighting")

# Uploaders
ckpt_file = st.file_uploader("Checkpoint (.pth)", type=["pth"])
lr_file = st.file_uploader("Low-Res Image", type=["png", "jpg", "jpeg"])
hr_file = st.file_uploader(
    "High-Res Image (optional, for LPIPS)", type=["png", "jpg", "jpeg"])

if not ckpt_file or not lr_file:
    st.warning("Please upload both a checkpoint and an LR image.")
    st.stop()

# Load images
lr_img = Image.open(lr_file).convert("RGB")
hr_img = Image.open(hr_file).convert("RGB") if hr_file else None

# Sidebar options
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {device}")

use_tta = st.sidebar.checkbox("Use TTA (×8)", value=False)

# fusion_scale is a global multiplier on the shallow/deep weight offsets. It lets you amplify or attenuate how strongly your chosen weights (shallow_weight, deep_weight) modify the fused features.
# fusion_scale = 1.0 - use exactly the offset implied by the weights, fusion_scale = 2.0 - double that effect, fusion_scale = 0.5 - halve that effect
fusion_scale = st.sidebar.slider("Fusion Scale", 0.1, 2.0, 1.0, 0.1)

approach = st.sidebar.radio("Weighting", ["Manual", "Local Search"], index=0)

if approach == "Manual":
    s_val = st.sidebar.slider("Shallow Weight", 0.5, 2.0, 1.0, 0.05)
    d_val = st.sidebar.slider("Deep Weight",    0.5, 2.0, 1.0, 0.05)
    best_w = BranchWeights(s_val, d_val)
else:
    steps = st.sidebar.slider("Search Steps", 1, 10, 5)
    step_size = st.sidebar.slider("Step Size",   0.01, 0.5, 0.1, 0.01)
    if hr_img is None:
        st.sidebar.warning("HR required for local search.")
        best_w = BranchWeights(1.0, 1.0)
    elif st.sidebar.button("Run Local Search"):
        # load model & search
        with st.spinner("Loading checkpoint..."):
            ckpt_bytes = ckpt_file.read()
            ckpt_buffer = io.BytesIO(ckpt_bytes)
            ckpt = torch.load(ckpt_buffer, map_location=device)
            model = DualBranch(
                scale_factor=config.scale_factor,
                in_channels=3,
                n_feats=config.num_feats,
                num_groups_shallow=config.num_groups_shallow,
                num_groups_deep=config.num_groups_deep,
                num_blocks_per_group=config.num_blocks_per_group,
                reduction=16
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
        proc = DualBranchProcessor(model, device)
        lr_t = proc.preprocess(lr_img)
        hr_t = proc.preprocess(hr_img)
        best_w = proc.local_search_lpips(
            lr_t, hr_t, fusion_scale, steps=steps, step_size=step_size)
        st.sidebar.success(
            f"Found weights S={best_w.shallow:.2f}, D={best_w.deep:.2f}")
    else:
        best_w = BranchWeights(1.0, 1.0)

# Run inference
if st.button("Process Image"):
    # load model once
    ckpt_bytes = ckpt_file.read()
    ckpt_buffer = io.BytesIO(ckpt_bytes)
    ckpt = torch.load(ckpt_buffer, map_location=device)
    model = DualBranch(
        scale_factor=config.scale_factor,
        in_channels=3,
        n_feats=config.num_feats,
        num_groups_shallow=config.num_groups_shallow,
        num_groups_deep=config.num_groups_deep,
        num_blocks_per_group=config.num_blocks_per_group,
        reduction=16
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    proc = DualBranchProcessor(model, device)
    lr_t = proc.preprocess(lr_img)

    # baseline
    with st.spinner("Baseline SR..."):
        base_t = proc.baseline_forward(lr_t, use_tta)
        base_np = proc.postprocess(base_t)
        st.session_state.base_np = base_np

    # weighted
    with st.spinner("Weighted SR..."):
        wtd_t = proc.weighted_forward(lr_t, best_w, fusion_scale, use_tta)
        wtd_np = proc.postprocess(wtd_t)
        st.session_state.wtd_np = wtd_np

    # display
    c1, c2 = st.columns(2)
    c1.subheader("Baseline")
    c1.image(base_np, use_container_width=True)
    c2.subheader(f"Weighted (S={best_w.shallow:.2f}, D={best_w.deep:.2f})")
    c2.image(wtd_np, use_container_width=True)

    # LPIPS if possible
    if hr_img:
        hr_np = np.array(hr_img)/255.0
        lp_b = calc_lpips(base_np, hr_np)
        lp_w = calc_lpips(wtd_np, hr_np)
        st.write(
            f"Baseline LPIPS: **{lp_b:.4f}**, Weighted LPIPS: **{lp_w:.4f}**, Δ: **{lp_w-lp_b:+.4f}**")

    # downloads
    buf_b = io.BytesIO()
    Image.fromarray((base_np*255).astype('uint8')).save(buf_b, format="PNG")
    buf_w = io.BytesIO()
    Image.fromarray((wtd_np*255).astype('uint8')).save(buf_w, format="PNG")
    d1, d2 = st.columns(2)
    d1.download_button("Download Baseline", buf_b.getvalue(),
                       "baseline.png", "image/png")
    d2.download_button("Download Weighted", buf_w.getvalue(),
                       "weighted.png", "image/png")
