# ================================================================
# utils.py  (オンザフライ読み込み版 / 中央クロップ対応)
# ================================================================
"""CD-SEM超解像プロジェクト用のユーティリティ関数とクラス群。

主な機能:
  - 乱数シード固定関数
  - パッチ間のサブピクセル位置合わせ関数
  - 画像の中央領域を切り出し、そこからオンザフライで画像パッチを読み込むDatasetクラス (SEMPatchDataset)
  - カスタム損失関数 (EdgeLoss, CDLoss)
"""
from __future__ import annotations
import cv2
import random
import os
import numpy as np
from pathlib import Path
from skimage.registration import phase_cross_correlation
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import config #プロジェクト共通設定

SUFFIX_LEN = 10  # ファイル名の末尾N文字でLR/HRペアをマッチング

# ------------------------------------------------------------
# 乱数シード固定
# ------------------------------------------------------------
def seed_everything(seed: int = 42):
    """各種乱数生成器のシードを固定して再現性を確保します。

    Args:
        seed (int, optional): 固定するシード値。デフォルトは42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ------------------------------------------------------------
# 前処理 (補正なし) & 位置合わせ (パッチ単位)
# ------------------------------------------------------------
def _align_patch(hr_patch: np.ndarray, lr_patch: np.ndarray) -> np.ndarray:
    """2つの画像パッチ間をサブピクセル精度で位置合わせします。

    Args:
        hr_patch (np.ndarray): 高解像度画像パッチ (基準)。
        lr_patch (np.ndarray): 低解像度画像パッチ (位置合わせ対象)。

    Returns:
        np.ndarray: 位置合わせされた低解像度画像パッチ。
    """
    if hr_patch.ndim != 2 or lr_patch.ndim != 2:
        raise ValueError("Input patches for alignment must be 2D (grayscale).")
    hr_patch_float = hr_patch.astype(np.float32)
    lr_patch_float = lr_patch.astype(np.float32)
    shift, error, phasediff = phase_cross_correlation(hr_patch_float, lr_patch_float, upsample_factor=20)
    tx = -shift[1]
    ty = -shift[0]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dsize = (hr_patch.shape[1], hr_patch.shape[0])
    aligned_lr_patch = cv2.warpAffine(lr_patch, M, dsize,
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
    return aligned_lr_patch

# ------------------------------------------------------------
# Dataset (オンザフライ読み込み & 中央クロップ)
# ------------------------------------------------------------
class SEMPatchDataset(Dataset):
    """CD-SEM画像のLR/HRペアから、まず画像中央の特定領域を切り出し、
    その領域内からオンザフライでランダムな正方形パッチを提供します。

    Attributes:
        patch_size (int): 切り出す正方形パッチの一辺の長さ。
        repeats (int): 1つの画像ペアから何回パッチをサンプリングするか。
        low_paths (list[Path]): LR画像のファイルパスのリスト。
        high_paths (list[Path]): HR画像のファイルパスのリスト。
        keys (list[str]): LR/HRペアをマッチングするための共通ファイル名末尾部分のリスト。
        central_crop_height (int): 切り出す中央領域の高さ。
        central_crop_width (int): 切り出す中央領域の幅。
    """
    def __init__(self,
                 low_dir: list[Path] | Path,
                 high_dir: list[Path] | Path,
                 patch: int = config.PATCH_SIZE, # configからデフォルト値を取得
                 stride: int = 64, # 現在の実装では未使用
                 repeats: int = 4,
                 central_crop_height: int = config.CENTRAL_CROP_HEIGHT,
                 central_crop_width: int = config.CENTRAL_CROP_WIDTH):
        """
        Args:
            low_dir (list[Path] | Path): LR画像が格納されたディレクトリ(群)。
            high_dir (list[Path] | Path): HR画像が格納されたディレクトリ(群)。
            patch (int, optional): 切り出す正方形パッチのサイズ。
            stride (int, optional): パッチ切り出し時のストライド (現在未使用)。
            repeats (int, optional): 1画像あたりのサンプリング回数。
            central_crop_height (int, optional): 切り出す中央領域の高さ。
            central_crop_width (int, optional): 切り出す中央領域の幅。
        """
        self.patch_size = patch
        self.repeats = repeats
        self._stride = stride # 未使用だが保持
        self.central_crop_height = central_crop_height
        self.central_crop_width = central_crop_width

        if self.patch_size > self.central_crop_width or self.patch_size > self.central_crop_height:
            print(f"Warning: PATCH_SIZE ({self.patch_size}) is larger than the central crop dimensions "
                  f"({self.central_crop_height}x{self.central_crop_width}). "
                  f"Patch will be clamped to the central crop region or may cause errors if not handled.")
            # 必要に応じて patch_size を調整するロジックを追加することも可能
            # self.patch_size = min(self.patch_size, self.central_crop_width, self.central_crop_height)


        def collect_files(dirs_config: list[Path] | Path) -> dict[str, Path]:
            import glob as _glob
            file_map: dict[str, Path] = {}
            dirs_to_scan = [dirs_config] if not isinstance(dirs_config, list) else dirs_config
            for d_path in dirs_to_scan:
                if not d_path.is_dir():
                    print(f"Warning: Directory not found: {d_path}")
                    continue
                pattern = str(d_path / f"{config.FILE_PREFIX}*.jpg")
                for p_str in _glob.glob(pattern):
                    p_obj = Path(p_str)
                    if p_obj.name.startswith(config.FILE_PREFIX):
                        if len(p_obj.name) >= SUFFIX_LEN:
                            file_map[p_obj.name[-SUFFIX_LEN:]] = p_obj
                        else:
                            print(f"Warning: Filename '{p_obj.name}' shorter than SUFFIX_LEN. Skipping.")
            return file_map

        low_map  = collect_files(low_dir)
        high_map = collect_files(high_dir)
        self.keys = sorted(list(set(low_map.keys()) & set(high_map.keys())))
        if not self.keys:
             raise AssertionError(f"No matching LR/HR image pairs found for prefix '{config.FILE_PREFIX}'. "
                                  f"Checked LR: {low_dir}, HR: {high_dir}")
        self.low_paths  = [low_map[k]  for k in self.keys]
        self.high_paths = [high_map[k] for k in self.keys]
        print(f"Found {len(self.keys)} matching image pairs. Using central crop {self.central_crop_height}x{self.central_crop_width}, patch size {self.patch_size}.")

    def __len__(self) -> int:
        return len(self.keys) * self.repeats

    def _get_central_crop(self, img: np.ndarray, filename: str) -> np.ndarray:
        """画像の中央部分を切り出します。"""
        H_orig, W_orig = img.shape[:2] # カラー画像も考慮して[:2]

        if H_orig < self.central_crop_height or W_orig < self.central_crop_width:
            # 画像が指定された中央クロップ領域より小さい場合は、画像をそのまま返し警告
            print(f"Warning: Image '{filename}' ({H_orig}x{W_orig}) is smaller than "
                  f"central crop size ({self.central_crop_height}x{self.central_crop_width}). "
                  f"Using the original image as the crop.")
            return img

        y_start = (H_orig - self.central_crop_height) // 2
        x_start = (W_orig - self.central_crop_width) // 2
        
        central_region = img[y_start : y_start + self.central_crop_height,
                             x_start : x_start + self.central_crop_width]
        return central_region

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """指定されたインデックスのデータ（LRパッチとHRパッチのペア）を取得します。

        画像を読み込み、中央領域を切り出し、その中からランダムパッチを生成します。
        """
        real_idx = idx // self.repeats
        lr_path = self.low_paths[real_idx]
        hr_path = self.high_paths[real_idx]

        try:
            lr_img_orig = cv2.imread(str(lr_path), cv2.IMREAD_GRAYSCALE)
            hr_img_orig = cv2.imread(str(hr_path), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            raise FileNotFoundError(f"Error reading image files: LR='{lr_path}', HR='{hr_path}'. Original error: {e}")

        if lr_img_orig is None: raise FileNotFoundError(f"Failed to load LR image: {lr_path}")
        if hr_img_orig is None: raise FileNotFoundError(f"Failed to load HR image: {hr_path}")

        # 中央領域を切り出す
        lr_central_crop = self._get_central_crop(lr_img_orig.astype(np.float32), lr_path.name)
        hr_central_crop = self._get_central_crop(hr_img_orig.astype(np.float32), hr_path.name)

        H_crop, W_crop = hr_central_crop.shape

        # パッチサイズがクロップ領域より大きい場合のハンドリング
        current_patch_size_h = min(self.patch_size, H_crop)
        current_patch_size_w = min(self.patch_size, W_crop)
        if self.patch_size > H_crop or self.patch_size > W_crop:
             print(f"Warning: PATCH_SIZE ({self.patch_size}) is larger than cropped region "
                   f"({H_crop}x{W_crop}) for {hr_path.name}. "
                   f"Clamping patch to {current_patch_size_h}x{current_patch_size_w}.")


        # クロップされた中央領域内からランダムなパッチ座標を生成
        if H_crop <= current_patch_size_h: # クロップ高 <= パッチ高 (パッチが高すぎる場合)
            y = 0
        else:
            y = random.randint(0, H_crop - current_patch_size_h)
        
        if W_crop <= current_patch_size_w: # クロップ幅 <= パッチ幅
            x = 0
        else:
            x = random.randint(0, W_crop - current_patch_size_w)

        hr_patch = hr_central_crop[y : y + current_patch_size_h, x : x + current_patch_size_w]
        lr_patch_candidate = lr_central_crop[y : y + current_patch_size_h, x : x + current_patch_size_w]
        
        lr_aligned_patch = _align_patch(hr_patch, lr_patch_candidate)

        if random.random() < 0.5: # ランダム左右反転
            hr_patch = hr_patch[:, ::-1].copy()
            lr_aligned_patch = lr_aligned_patch[:, ::-1].copy()

        hr_tensor = torch.from_numpy(hr_patch / 127.5 - 1.0).unsqueeze(0).float()
        lr_tensor = torch.from_numpy(lr_aligned_patch / 127.5 - 1.0).unsqueeze(0).float()
        
        return lr_tensor, hr_tensor

# ------------------------------------------------------------
# 損失関数 (変更なし)
# ------------------------------------------------------------
class EdgeLoss(torch.nn.Module):
    """画像の勾配マップ（エッジ）間のL1損失を計算するクラス。"""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        gx_sr = F.conv2d(sr, self.kx, padding='same'); gx_hr = F.conv2d(hr, self.kx, padding='same')
        gy_sr = F.conv2d(sr, self.ky, padding='same'); gy_hr = F.conv2d(hr, self.ky, padding='same')
        return torch.mean(torch.abs(gx_sr - gx_hr)) + torch.mean(torch.abs(gy_sr - gy_hr))

class CDLoss(torch.nn.Module):
    """CD (Critical Dimension) 計測に関連する簡易的な損失関数。"""
    def __init__(self):
        super().__init__()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        diff_y_sr = torch.abs(sr[:,:,1:,:] - sr[:,:,:-1,:])
        diff_y_hr = torch.abs(hr[:,:,1:,:] - hr[:,:,:-1,:])
        return torch.mean(torch.abs(diff_y_sr - diff_y_hr))

