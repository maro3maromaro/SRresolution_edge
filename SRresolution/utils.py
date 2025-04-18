# ================================================================
# utils.py  (前処理・Dataset・損失関数ユーティリティ)
# ================================================================
#   構成
#   ① 乱数シード固定          : seed_everything()
#   ② 前処理ラッパー          : apply_corr()  ← フラット / ダークが無い場合はスルー
#   ③ サブピクセル位置合わせ   : register_subpixel()
#   ④ Dataset                : SEMPatchDataset (白黒ペア → パッチ)
#   ⑤ 損失関数                : EdgeLoss / CDLoss
# ------------------------------------------------
#   ※ ユーザ環境では "ダーク画像なし & フラットなし" なので、
#      補正が不要の場合はそのまま画像を返す仕様に変更しました。
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ------------------------------------------------------------
# ② 前処理ラッパー (フラット / ダークが無ければスルー)
# ------------------------------------------------------------

def apply_corr(img: np.ndarray, flat: np.ndarray | None = None, dark: np.ndarray | None = None) -> np.ndarray:
    """フラット / ダーク補正を行う。
        flat が None の場合は補正せず float32 に変換して返す。
    """
    if flat is None:
        return img.astype(np.float32)  # 無補正

    # --- フラットのみ or ダーク込み補正 ---
    img_f  = img.astype(np.float32)
    if dark is None:
        flat_f = flat.astype(np.float32)
        corr = img_f / (flat_f + 1e-6) * np.mean(flat_f)
    else:
        flat_f = flat.astype(np.float32)
        dark_f = dark.astype(np.float32)
        corr = (img_f - dark_f) / (flat_f - dark_f + 1e-6) * np.mean(flat_f - dark_f)
    return np.clip(corr, 0, 255)

# ------------------------------------------------------------
# ③ サブピクセル位置合わせ (Phase Only Correlation)
# ------------------------------------------------------------

def register_subpixel(ref: np.ndarray, mov: np.ndarray, up: int = 100) -> np.ndarray:
    shift, *_ = phase_cross_correlation(ref, mov, upsample_factor=up)
    M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
    return cv2.warpAffine(mov, M, ref.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)

# ------------------------------------------------------------
# ④ Dataset: HR/LR ペア → パッチ
# ------------------------------------------------------------
class SEMPatchDataset(Dataset):
    """SEM 低画質・高画質ペアを読み込み、パッチ単位で返す Dataset"""

    def __init__(self,
                 low_dir: str | Path,
                 high_dir: str | Path,
                 flat_path: str | Path | None = None,
                 dark_path: str | Path | None = None,
                 patch: int = 256,
                 stride: int = 64):
        self.low_paths  = sorted(Path(low_dir).glob('*.jpg'))
        self.high_paths = sorted(Path(high_dir).glob('*.jpg'))
        assert len(self.low_paths)==len(self.high_paths), 'LR/HR の枚数が一致しません'

        # 補正画像 (存在しなければ None)
        self.flat = cv2.imread(str(flat_path),0).astype(np.float32) if flat_path and Path(flat_path).exists() else None
        self.dark = cv2.imread(str(dark_path),0).astype(np.float32) if dark_path and Path(dark_path).exists() else None

        self.patch, self.stride = patch, stride

        self.low_imgs, self.high_imgs = [], []
        for lp,hp in zip(self.low_paths,self.high_paths):
            l_raw = cv2.imread(str(lp),0)
            h_raw = cv2.imread(str(hp),0)
            l_corr = apply_corr(l_raw, self.flat, self.dark)
            h_corr = apply_corr(h_raw, self.flat, self.dark)
            l_reg  = register_subpixel(h_corr, l_corr)
            h_roi, l_roi = self._crop_common(h_corr,l_reg)
            self.high_imgs.append(h_roi)
            self.low_imgs.append(l_roi)

        # 全パッチ座標列挙
        self.coords = [(idx,y,x)
                        for idx,h in enumerate(self.high_imgs)
                        for y in range(0,h.shape[0]-patch+1,stride)
                        for x in range(0,h.shape[1]-patch+1,stride)]

    @staticmethod
    def _crop_common(a,b):
        h=min(a.shape[0],b.shape[0]); w=min(a.shape[1],b.shape[1])
        return a[:h,:w], b[:h,:w]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx:int):
        img_idx,y,x=self.coords[idx]
        h=self.high_imgs[img_idx][y:y+self.patch, x:x+self.patch]
        l=self.low_imgs [img_idx][y:y+self.patch, x:x+self.patch]
        if random.random()<0.5:
            h,l = h[:,::-1], l[:,::-1]
        return (torch.from_numpy(l/127.5-1).unsqueeze(0).float(),
                torch.from_numpy(h/127.5-1).unsqueeze(0).float())

# ------------------------------------------------------------
# ⑤ 損失関数
# ------------------------------------------------------------
class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx',kx)
        self.register_buffer('ky',kx.transpose(2,3))
    def forward(self,sr,hr):
        gx_sr=F.conv2d(sr,self.kx,padding=1); gy_sr=F.conv2d(sr,self.ky,padding=1)
        gx_hr=F.conv2d(hr,self.kx,padding=1); gy_hr=F.conv2d(hr,self.ky,padding=1)
        return torch.mean(torch.abs(gx_sr-gx_hr)+torch.abs(gy_sr-gy_hr))

class CDLoss(torch.nn.Module):
    def forward(self,sr,hr):
        gy_sr=torch.abs(sr[:,:,1:,:]-sr[:,:,:-1,:])
        gy_hr=torch.abs(hr[:,:,1:,:]-hr[:,:,:-1,:])
        return torch.mean(torch.abs(gy_sr-gy_hr))
