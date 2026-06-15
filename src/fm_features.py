# -*- coding: utf-8 -*-
"""ClinScope FM 引擎共用模組: Phikon-v2(病理 ViT-L, frozen)特徵抽取。
train_fm_probe.py(訓練) 與 app.py(推論) 共用,確保訓練/推論前處理一致。
作法: 影像 → resize (grid*224) → grid×grid 個 224 tile → 各取 CLS token(1024維) → mean-pool。
grid=3 給大圖(WSI 縮圖);grid=1 給已是 patch 級的小圖。"""
import numpy as np
import torch
from PIL import Image

MODEL_NAME = "owkin/phikon-v2"
TILE = 224
_state = {"model": None, "mean": None, "std": None, "device": None}


def load_phikon(device):
    """載入並快取 Phikon-v2(frozen)。device: torch.device 或 'cuda'/'cpu'。"""
    if _state["model"] is None:
        from transformers import AutoModel, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        _state.update(
            model=model, device=device,
            mean=torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device),
            std=torch.tensor(proc.image_std).view(1, 3, 1, 1).to(device))
    return _state["model"]


def _to_tiles(pil_img, grid):
    im = pil_img.convert("RGB").resize((grid * TILE, grid * TILE), Image.BILINEAR)
    arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0  # 3,H,W (np.array=可寫副本)
    return [arr[:, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
            for r in range(grid) for c in range(grid)]


@torch.no_grad()
def extract_feature(pil_img, device, grid=3):
    """單張 → 1024 維特徵(推論用)。"""
    load_phikon(device)
    t = torch.stack(_to_tiles(pil_img, grid)).to(device)
    t = (t - _state["mean"]) / _state["std"]
    cls = load_phikon(device)(pixel_values=t).last_hidden_state[:, 0]  # g²,1024
    return cls.mean(0).cpu().numpy()


@torch.no_grad()
def extract_batch(pil_imgs, device, grid=3, tile_bs=128):
    """一批影像 → (N,1024)(訓練用,跨影像批次化 tile forward 加速)。"""
    model = load_phikon(device)
    feats = []
    all_tiles, owner = [], []
    for i, im in enumerate(pil_imgs):
        ts = _to_tiles(im, grid)
        all_tiles += ts
        owner += [i] * len(ts)
    all_tiles = torch.stack(all_tiles)            # (N*g², 3,224,224)
    owner = np.array(owner)
    embs = []
    for s in range(0, len(all_tiles), tile_bs):
        t = all_tiles[s:s + tile_bs].to(device)
        t = (t - _state["mean"]) / _state["std"]
        embs.append(model(pixel_values=t).last_hidden_state[:, 0].cpu())
    embs = torch.cat(embs).numpy()                # (N*g²,1024)
    for i in range(len(pil_imgs)):
        feats.append(embs[owner == i].mean(0))
    return np.stack(feats)
