# ================================================================
# utils.py  (前処理・Dataset・損失関数ユーティリティ)
# ================================================================
#   * CD‑SEM 超解像パイプラインの補助関数を集約
#   * ① 乱数シード固定               : seed_everything()
#   * ② フラット補正 (ダーク画像なし): apply_flat() / apply_dark_flat()
#   * ③ サブピクセル位置合わせ        : register_subpixel()
#   * ④ データセット (パッチ抽出)     : SEMPatchDataset
#   * ⑤ ライン保持用損失関数          : EdgeLoss / CDLoss
# ------------------------------------------------
#  **変更点**
#   - ダーク画像が用意できない環境へ対応。
#   - dark_path が None もしくはファイル無しの場合は "ダーク無しフラット補正" を行う。
# ================================================================

from __future__ import annotations
import cv2, random, os
import numpy as np
from pathlib import Path
from skimage.registration import phase_cross_correlation
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# ------------------------------------------------------------
# ① 乱数シード固定
# ------------------------------------------------------------

def seed_everything(seed: int = 42):
    """Python / NumPy / PyTorch の乱数シードを統一設定して再現性を確保"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ------------------------------------------------------------
# ② フラット補正 (ダーク画像なし対応)
# ------------------------------------------------------------

def apply_flat(img: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """フラットフィールドのみでムラを補正 (dark なし)
    (img / flat) * mean(flat) 方式で輝度を正規化
    """
    img_f  = img.astype(np.float32)
    flat_f = flat.astype(np.float32)
    corr = img_f / (flat_f + 1e-6) * np.mean(flat_f)
    return np.clip(corr, 0, 255)


def apply_dark_flat(img: np.ndarray, flat: np.ndarray, dark: np.ndarray | None = None) -> np.ndarray:
    """ダーク画像の有無を判定して補正方式を切替えるラッパー

    Args:
        img  : Original SEM image (0‑255)
        flat : Flat‑field image (同サイズ)
        dark : Dark image or None
    """
    if dark is None or dark.size == 0:
        return apply_flat(img, flat)
    # --- dark あり補正 ---
    img_f  = img.astype(np.float32)
    dark_f = dark.astype(np.float32)
    flat_f = flat.astype(np.float32)
    corr = (img_f - dark_f) / (flat_f - dark_f + 1e-6) * np.mean(flat_f - dark_f)
    return np.clip(corr, 0, 255)

# ------------------------------------------------------------
# ③ サブピクセル位置合わせ (Phase‑Only Correlation)
# ------------------------------------------------------------

def register_subpixel(ref: np.ndarray, mov: np.ndarray, up: int = 100) -> np.ndarray:
    """低画質 mov を ref に sub‑px 精度で整列し、warpAffine で補正"""
    shift, *_ = phase_cross_correlation(ref, mov, upsample_factor=up)
    matrix = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
    return cv2.warpAffine(mov, matrix, ref.shape[::-1],
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)

# ------------------------------------------------------------
# ④ Dataset: 低 / 高画像ペア → パッチ列挙
# ------------------------------------------------------------
class SEMPatchDataset(Dataset):
    """SEM 画像ペアを読み込み、補正・位置合わせ後にパッチを提供する Dataset"""

    def __init__(self,
                 low_dir: str | Path,
                 high_dir: str | Path,
                 flat_path: str | Path,
                 dark_path: str | Path | None = None,
                 patch: int = 256,
                 stride: int = 64):
        # ---------- 画像パス列挙 ----------
        self.low_paths  = sorted(Path(low_dir).glob('*.jpg'))
        self.high_paths = sorted(Path(high_dir).glob('*.jpg'))
        assert len(self.low_paths) == len(self.high_paths), "低画質と高画質の枚数が一致しません"

        # ---------- 補正用画像読み込み ----------
        self.flat = cv2.imread(str(flat_path), 0).astype(np.float32)
        # dark が無い場合は空 array
        self.dark = None
        if dark_path is not None and Path(dark_path).exists():
            self.dark = cv2.imread(str(dark_path), 0).astype(np.float32)

        self.patch  = patch
        self.stride = stride

        # ---------- 全画像前処理 ----------
        self.low_imgs, self.high_imgs = [], []
        for lp, hp in zip(self.low_paths, self.high_paths):
            l_raw = cv2.imread(str(lp), 0)
            h_raw = cv2.imread(str(hp), 0)
            # フラット / ダーク補正
            l_corr = apply_dark_flat(l_raw, self.flat, self.dark)
            h_corr = apply_dark_flat(h_raw, self.flat, self.dark)
            # 位置合わせ
            l_reg  = register_subpixel(h_corr, l_corr)
            # 共通 ROI
            h_roi, l_roi = self._crop_common(h_corr, l_reg)
            self.high_imgs.append(h_roi)
            self.low_imgs.append(l_roi)

        # ---------- パッチ座標列挙 ----------
        self.coords: list[tuple[int,int,int]] = []
        for idx, h in enumerate(self.high_imgs):
            for y in range(0, h.shape[0] - patch + 1, stride):
                for x in range(0, h.shape[1] - patch + 1, stride):
                    self.coords.append((idx, y, x))

    # ---- 共通 ROI 抽出 ----
    @staticmethod
    def _crop_common(a: np.ndarray, b: np.ndarray):
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        return a[:h, :w], b[:h, :w]

    # ---- Dataset API ----
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx: int):
        img_idx, y, x = self.coords[idx]
        h_patch = self.high_imgs[img_idx][y:y+self.patch, x:x+self.patch]
        l_patch = self.low_imgs [img_idx][y:y+self.patch, x:x+self.patch]

        # 左右フリップ拡張
        if random.random() < 0.5:
            h_patch = h_patch[:, ::-1]
            l_patch = l_patch[:, ::-1]

        # [-1,1] 正規化 + Tensor 化 (C,H,W)
        h_tensor = torch.from_numpy(h_patch/127.5 - 1.).unsqueeze(0).float()
        l_tensor = torch.from_numpy(l_patch/127.5 - 1.).unsqueeze(0).float()
        return l_tensor, h_tensor

# ------------------------------------------------------------
# ⑤ 損失関数 (EdgeLoss / CDLoss)
# ------------------------------------------------------------
class EdgeLoss(torch.nn.Module):
    """Sobel 勾配差を L1 で評価し、エッジ形状を保持"""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
    def forward(self, sr, hr):
        gx_sr = F.conv2d(sr, self.kx, padding=1); gy_sr = F.conv2d(sr, self.ky, padding=1)
        gx_hr = F.conv2d(hr, self.kx, padding=1); gy_hr = F.conv2d(hr, self.ky, padding=1)
        return torch.mean(torch.abs(gx_sr-gx_hr) + torch.abs(gy_sr-gy_hr))

class CDLoss(torch.nn.Module):
    """一次差分の L1 差でライン幅変動を抑える損失"""
    def forward(self, sr, hr):
        gy_sr = torch.abs(sr[:,:,1:,:] - sr[:,:,:-1,:])
        gy_hr = torch.abs(hr[:,:,1:,:] - hr[:,:,:-1,:])
        return torch.mean(torch.abs(gy_sr - gy_hr))
