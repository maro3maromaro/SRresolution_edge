# ================================================================
# utils.py  (前処理・Dataset・損失関数) ★複数ディレクトリ対応版★
# ================================================================
#  概要:
#    - config.TRAIN_LOW_DIRS / TRAIN_HIGH_DIRS に列挙された複数フォルダから
#      ファイル名が config.FILE_PREFIX で始まる JPG を収集
#    - フラット／ダーク補正はオプション (今回は None でスキップ)
#    - ペア画像は "同名ファイル" のみ残し、欠損は自動で除外
#    - サブピクセル登録後、パッチに分割して学習用に返す
# ================================================================
from __future__ import annotations
import cv2, random, os
import numpy as np
from pathlib import Path
from skimage.registration import phase_cross_correlation
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import config  # ★ コンフィグを読み込む

# ------------------------------------------------------------
# 乱数シード固定
# ------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ------------------------------------------------------------
# 簡易 前処理 (今回: 補正なし → float32 にキャストのみ)
# ------------------------------------------------------------

def apply_corr(img: np.ndarray) -> np.ndarray:
    """補正をスキップし、そのまま float32 に変換"""
    return img.astype(np.float32)

# ------------------------------------------------------------
# サブピクセル位置合わせ
# ------------------------------------------------------------

def register_subpixel(ref: np.ndarray, mov: np.ndarray, up: int = 100) -> np.ndarray:
    shift, *_ = phase_cross_correlation(ref, mov, upsample_factor=up)
    M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
    return cv2.warpAffine(mov, M, ref.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class SEMPatchDataset(Dataset):
    """SEM LR/HR ペア画像を前処理し、ランダムパッチを返す Dataset"""

    def __init__(self,
                 low_dir: Path | list[Path],
                 high_dir: Path | list[Path],
                 patch: int = 256,
                 stride: int = 64):
        # --- ディレクトリをリスト化 ---
        low_dirs  = [low_dir]  if isinstance(low_dir,  Path) else list(low_dir)
        high_dirs = [high_dir] if isinstance(high_dir, Path) else list(high_dir)

        # --- ファイル収集 (prefix でフィルタ) ---
        def collect(dirs):
            return {p.name: p for dir_ in dirs for p in dir_.glob('*.jpg') if p.name.startswith(config.FILE_PREFIX)}

        low_dict  = collect(low_dirs)
        high_dict = collect(high_dirs)

        # --- 同名ペアのみ残す ---
        common_keys = sorted(set(low_dict) & set(high_dict))
        self.low_paths  = [low_dict[k]  for k in common_keys]
        self.high_paths = [high_dict[k] for k in common_keys]
        assert self.low_paths, '一致する LR/HR ペアが見つかりません'

        self.patch  = patch
        self.stride = stride

        # --- 画像ロード & 前処理 & 位置合わせ ---
        self.low_imgs, self.high_imgs = [], []
        for lp, hp in zip(self.low_paths, self.high_paths):
            l_raw = cv2.imread(str(lp), 0)
            h_raw = cv2.imread(str(hp), 0)
            l = apply_corr(l_raw)
            h = apply_corr(h_raw)
            l_reg = register_subpixel(h, l)
            h_roi, l_roi = self._crop_common(h, l_reg)
            self.high_imgs.append(h_roi)
            self.low_imgs.append(l_roi)

        # --- パッチ座標リスト ---
        self.coords = [(idx, y, x)
                       for idx, h in enumerate(self.high_imgs)
                       for y in range(0, h.shape[0]-patch+1, stride)
                       for x in range(0, h.shape[1]-patch+1, stride)]

    @staticmethod
    def _crop_common(a, b):
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        return a[:h, :w], b[:h, :w]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        img_idx, y, x = self.coords[idx]
        h = self.high_imgs[img_idx][y:y+self.patch, x:x+self.patch]
        l = self.low_imgs [img_idx][y:y+self.patch, x:x+self.patch]
        if random.random() < 0.5:
            h, l = h[:, ::-1], l[:, ::-1]
        h_t = torch.from_numpy(h/127.5 - 1.).unsqueeze(0).float()
        l_t = torch.from_numpy(l/127.5 - 1.).unsqueeze(0).float()
        return l_t, h_t

# ------------------------------------------------------------
# 損失関数 (Edge / CD)
# ------------------------------------------------------------
class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', kx.transpose(2,3))
    def forward(self, sr, hr):
        gx_sr = F.conv2d(sr, self.kx, padding=1); gy_sr = F.conv2d(sr, self.ky, padding=1)
        gx_hr = F.conv2d(hr, self.kx, padding=1); gy_hr = F.conv2d(hr, self.ky, padding=1)
        return torch.mean(torch.abs(gx_sr-gx_hr) + torch.abs(gy_sr-gy_hr))

class CDLoss(torch.nn.Module):
    def forward(self, sr, hr):
        gy_sr = torch.abs(sr[:,:,1:,:] - sr[:,:,:-1,:])
        gy_hr = torch.abs(hr[:,:,1:,:] - hr[:,:,:-1,:])
        return torch.mean(torch.abs(gy_sr - gy_hr))
