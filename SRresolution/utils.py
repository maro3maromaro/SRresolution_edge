# ================================================================
# utils.py  (オンザフライ読み込み版)  ★低メモリ対応★
# ================================================================
#   変更点:
#     • 画像をメモリにプリロードしない → CPU/RAM を節約
#     • __getitem__() で都度 jpg を読み込み & ランダムパッチ
#     • register_subpixel は画像全体ではなくパッチ領域で実行し高速化
# ================================================================
from __future__ import annotations
import cv2, random, os
import numpy as np
from pathlib import Path
from skimage.registration import phase_cross_correlation
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import config

SUFFIX_LEN = 10  # 末尾 N 文字で LR/HR をマッチング

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
# 前処理 (補正なし) & 位置合わせ (パッチ単位)
# ------------------------------------------------------------

def _align_patch(hr_patch: np.ndarray, lr_patch: np.ndarray) -> np.ndarray:
    """パッチ同士をサブピクセル精度で位置合わせ"""
    shift,_ = phase_cross_correlation(hr_patch, lr_patch, upsample_factor=20)
    M = np.float32([[1,0,-shift[1]],[0,1,-shift[0]]])
    return cv2.warpAffine(lr_patch, M, hr_patch.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)

# ------------------------------------------------------------
# Dataset (オンザフライ読み込み)
# ------------------------------------------------------------
class SEMPatchDataset(Dataset):
    """各ペアからランダムパッチを切り出すオンライン Dataset"""

    def __init__(self,
                 low_dir: list[Path] | Path,
                 high_dir: list[Path] | Path,
                 patch: int = 128,
                 stride: int = 128,
                 repeats: int = 4):
        self.patch = patch
        self.repeats = repeats  # 1 画像あたり何パッチ相当として扱うか

        # ----- ファイルペア収集 -----
        def collect(dirs):
            import glob as _glob
            mp={}
            for d in (dirs if isinstance(d,list) else [dirs]):
                for p in _glob.glob(str(Path(d)/'*.jpg')):
                    name=Path(p).name
                    if not name.startswith(config.FILE_PREFIX):
                        continue
                    mp[name[-SUFFIX_LEN:]]=Path(p)
            return mp
        low_map  = collect(low_dir)
        high_map = collect(high_dir)
        self.keys = sorted(set(low_map)&set(high_map))
        self.low_paths  = [low_map[k]  for k in self.keys]
        self.high_paths = [high_map[k] for k in self.keys]
        assert self.keys, 'No matching LR/HR pairs'

    def __len__(self):
        return len(self.keys) * self.repeats

    def __getitem__(self, idx):
        real_idx = idx // self.repeats
        lr_path = self.low_paths[real_idx]
        hr_path = self.high_paths[real_idx]

        lr_img = cv2.imread(str(lr_path), 0).astype(np.float32)
        hr_img = cv2.imread(str(hr_path), 0).astype(np.float32)

        # ランダム座標
        H,W = hr_img.shape
        y = random.randint(0, H - self.patch)
        x = random.randint(0, W - self.patch)

        hr_patch = hr_img[y:y+self.patch, x:x+self.patch]
        lr_patch = lr_img[y:y+self.patch, x:x+self.patch]
        lr_patch = _align_patch(hr_patch, lr_patch)

        # ランダム左右反転
        if random.random()<0.5:
            hr_patch = hr_patch[:, ::-1]
            lr_patch = lr_patch[:, ::-1]

        # [-1,1] 正規化
        hr_t = torch.from_numpy(hr_patch/127.5 - 1.).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr_patch/127.5 - 1.).unsqueeze(0).float()
        return lr_t, hr_t

# ------------------------------------------------------------
# 損失関数
# ------------------------------------------------------------
class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', kx.transpose(2,3))
    def forward(self,sr,hr):
        gx_sr = F.conv2d(sr,self.kx,padding=1); gy_sr = F.conv2d(sr,self.ky,padding=1)
        gx_hr = F.conv2d(hr,self.kx,padding=1); gy_hr = F.conv2d(hr,self.ky,padding=1)
        return torch.mean(torch.abs(gx_sr-gx_hr)+torch.abs(gy_sr-gy_hr))

class CDLoss(torch.nn.Module):
    def forward(self,sr,hr):
        gy_sr=torch.abs(sr[:,:,1:,:]-sr[:,:,:-1,:])
        gy_hr=torch.abs(hr[:,:,1:,:]-hr[:,:,:-1,:])
        return torch.mean(torch.abs(gy_sr-gy_hr))
