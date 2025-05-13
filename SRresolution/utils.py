# ================================================================
# utils.py  (オンザフライ読み込み版)  ★低メモリ対応★
# ================================================================
"""CD-SEM超解像プロジェクト用のユーティリティ関数とクラス群。

主な機能:
  - 乱数シード固定関数
  - パッチ間のサブピクセル位置合わせ関数
  - オンザフライで画像パッチを読み込むDatasetクラス (SEMPatchDataset)
  - カスタム損失関数 (EdgeLoss, CDLoss)
"""
from __future__ import annotations # for type hinting Path in older Python
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
    # torch.backends.cudnn.deterministic = True # これを入れると遅くなることがある
    # torch.backends.cudnn.benchmark = False   # 同上

# ------------------------------------------------------------
# 前処理 (補正なし) & 位置合わせ (パッチ単位)
# ------------------------------------------------------------

def _align_patch(hr_patch: np.ndarray, lr_patch: np.ndarray) -> np.ndarray:
    """2つの画像パッチ間をサブピクセル精度で位置合わせします。

    `phase_cross_correlation` を使用してズレを検出し、
    `lr_patch` を `hr_patch` に合わせてアフィン変換します。

    Args:
        hr_patch (np.ndarray): 高解像度画像パッチ (基準)。
        lr_patch (np.ndarray): 低解像度画像パッチ (位置合わせ対象)。

    Returns:
        np.ndarray: 位置合わせされた低解像度画像パッチ。
    """
    # Ensure inputs are grayscale (2D)
    if hr_patch.ndim != 2 or lr_patch.ndim != 2:
        raise ValueError("Input patches for alignment must be 2D (grayscale).")
        
    # skimage.registration.phase_cross_correlation expects non-zero images.
    # Add a small epsilon if images can be all zeros, though typically SEM images aren't.
    # For robustness, ensure images are float type for phase_cross_correlation
    hr_patch_float = hr_patch.astype(np.float32)
    lr_patch_float = lr_patch.astype(np.float32)

    # upsample_factorを大きくすると精度が上がるが計算時間も増える
    shift, error, phasediff = phase_cross_correlation(hr_patch_float, lr_patch_float, upsample_factor=20)
    
    # `shift` は (row_shift, col_shift) の順
    # `warpAffine` の変換行列Mのtx, tyは (col_shift, row_shift) の順で、
    # かつ `WARP_INVERSE_MAP` を使う場合、hr_patchからlr_patchへの逆変換なので符号が逆になる
    # M = [[1, 0, shift_x], [0, 1, shift_y]]
    # ここでは、lr_patch を hr_patch に合わせるので、lr_patch を (-shift[1], -shift[0]) だけ動かす
    tx = -shift[1] # col_shift (x方向のズレ)
    ty = -shift[0] # row_shift (y方向のズレ)
    
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # `dsize` は (width, height) の順
    dsize = (hr_patch.shape[1], hr_patch.shape[0]) 
    
    aligned_lr_patch = cv2.warpAffine(lr_patch, M, dsize, 
                                      flags=cv2.INTER_LINEAR, # WARP_INVERSE_MAPは不要、Mが順方向のため
                                      borderMode=cv2.BORDER_REPLICATE)
    return aligned_lr_patch

# ------------------------------------------------------------
# Dataset (オンザフライ読み込み)
# ------------------------------------------------------------
class SEMPatchDataset(Dataset):
    """CD-SEM画像の低解像度(LR)と高解像度(HR)のペアから、
    オンザフライでランダムなパッチを切り出して提供するDatasetクラス。

    画像はメモリにプリロードせず、`__getitem__`呼び出し時に都度読み込みます。
    LR/HRパッチ間のサブピクセル位置合わせも行います。

    Attributes:
        patch_size (int): 切り出すパッチのサイズ (一辺の長さ)。
        repeats (int): 1つの画像ペアから何回パッチをサンプリングするか (データ拡張の一種)。
        low_paths (list[Path]): LR画像のファイルパスのリスト。
        high_paths (list[Path]): HR画像のファイルパスのリスト。
        keys (list[str]): LR/HRペアをマッチングするための共通ファイル名末尾部分のリスト。
    """

    def __init__(self,
                 low_dir: list[Path] | Path,
                 high_dir: list[Path] | Path,
                 patch: int = 128,
                 stride: int = 128, # 現在の実装では未使用 (ランダムクロップのため)
                 repeats: int = 4):
        """
        Args:
            low_dir (list[Path] | Path): LR画像が格納されたディレクトリ(単一またはリスト)。
            high_dir (list[Path] | Path): HR画像が格納されたディレクトリ(単一またはリスト)。
            patch (int, optional): 切り出すパッチのサイズ。デフォルトは128。
            stride (int, optional): パッチ切り出し時のストライド。
                現在のランダムクロップ実装では未使用。デフォルトは128。
            repeats (int, optional): 1画像あたりのサンプリング回数。デフォルトは4。
        
        Raises:
            AssertionError: マッチするLR/HR画像ペアが見つからない場合。
        """
        self.patch_size = patch # patch引数名をpatch_sizeに変更して明確化
        self.repeats = repeats
        self._stride = stride # 未使用だが保持

        def collect_files(dirs_config: list[Path] | Path) -> dict[str, Path]:
            """指定されたディレクトリ(群)から画像ファイルを収集し、
            ファイル名末尾をキーとする辞書を作成します。
            """
            import glob as _glob
            file_map: dict[str, Path] = {}
            
            if not isinstance(dirs_config, list):
                dirs_to_scan = [dirs_config]
            else:
                dirs_to_scan = dirs_config
            
            for d_path in dirs_to_scan:
                if not d_path.is_dir():
                    print(f"Warning: Directory not found or not a directory: {d_path}")
                    continue
                # config.FILE_PREFIX を考慮したglobパターン
                pattern = str(d_path / f"{config.FILE_PREFIX}*.jpg")
                for p_str in _glob.glob(pattern):
                    p_obj = Path(p_str)
                    # ファイル名が本当に接頭辞で始まっているか再確認 (globのワイルドカード挙動のため)
                    if p_obj.name.startswith(config.FILE_PREFIX):
                         # SUFFIX_LEN がファイル名長を超える場合のエラーを回避
                        if len(p_obj.name) >= SUFFIX_LEN:
                            file_map[p_obj.name[-SUFFIX_LEN:]] = p_obj
                        else:
                            print(f"Warning: Filename '{p_obj.name}' is shorter than SUFFIX_LEN ({SUFFIX_LEN}). Skipping.")
            return file_map

        low_map  = collect_files(low_dir)
        high_map = collect_files(high_dir)
        
        self.keys = sorted(list(set(low_map.keys()) & set(high_map.keys())))
        
        if not self.keys:
             err_msg = "No matching LR/HR image pairs found. "
             err_msg += f"LR Dirs: {low_dir}, HR Dirs: {high_dir}, Prefix: {config.FILE_PREFIX}, SuffixLen: {SUFFIX_LEN}"
             raise AssertionError(err_msg)
             
        self.low_paths  = [low_map[k]  for k in self.keys]
        self.high_paths = [high_map[k] for k in self.keys]
        print(f"Found {len(self.keys)} matching image pairs for SEMPatchDataset.")


    def __len__(self) -> int:
        """データセットの総サンプル数を返します。

        画像ペア数 × repeats となります。
        """
        return len(self.keys) * self.repeats

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """指定されたインデックスのデータ（LRパッチとHRパッチのペア）を取得します。

        オンザフライで画像を読み込み、ランダムな位置からパッチを切り出し、
        位置合わせとデータ拡張（左右反転）を行い、テンソルに変換して返します。

        Args:
            idx (int): データサンプルのインデックス。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - lr_tensor (torch.Tensor): 低解像度画像パッチ (1, patch_size, patch_size)。[-1, 1]に正規化。
                - hr_tensor (torch.Tensor): 高解像度画像パッチ (1, patch_size, patch_size)。[-1, 1]に正規化。
        
        Raises:
            FileNotFoundError: 画像ファイルが見つからない場合。
            ValueError: パッチサイズが画像サイズより大きい場合。
        """
        real_idx = idx // self.repeats
        lr_path = self.low_paths[real_idx]
        hr_path = self.high_paths[real_idx]

        try:
            # cv2.IMREAD_GRAYSCALE (0) を明示
            lr_img = cv2.imread(str(lr_path), cv2.IMREAD_GRAYSCALE)
            hr_img = cv2.imread(str(hr_path), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            raise FileNotFoundError(f"Error reading image files: LR='{lr_path}', HR='{hr_path}'. Original error: {e}")

        if lr_img is None:
            raise FileNotFoundError(f"Failed to load LR image: {lr_path}")
        if hr_img is None:
            raise FileNotFoundError(f"Failed to load HR image: {hr_path}")
        
        # 型をfloat32に変換 (アライメントや正規化のため)
        lr_img = lr_img.astype(np.float32)
        hr_img = hr_img.astype(np.float32)

        H, W = hr_img.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Image size ({H}x{W}) is smaller than patch size ({self.patch_size}x{self.patch_size}) for {hr_path.name}")

        # ランダム座標でパッチをクロップ
        y = random.randint(0, H - self.patch_size)
        x = random.randint(0, W - self.patch_size)

        hr_patch = hr_img[y : y + self.patch_size, x : x + self.patch_size]
        # LR画像も同じ座標からクロップするが、HRとのズレがあるため、アライメント後にHRの範囲に合わせる
        # アライメントはHRパッチを基準に行うため、LRもHRと同じサイズでクロップしておく
        lr_patch_candidate = lr_img[y : y + self.patch_size, x : x + self.patch_size]
        
        # パッチ単位での位置合わせ
        lr_aligned_patch = _align_patch(hr_patch, lr_patch_candidate)

        # ランダム左右反転 (データ拡張)
        if random.random() < 0.5:
            hr_patch = hr_patch[:, ::-1].copy() # .copy() で連続性を保証
            lr_aligned_patch = lr_aligned_patch[:, ::-1].copy()

        # [-1, 1] に正規化し、テンソルに変換 (チャネル次元追加 C, H, W)
        # 画像の値域が0-255であることを想定
        hr_tensor = torch.from_numpy(hr_patch / 127.5 - 1.0).unsqueeze(0).float()
        lr_tensor = torch.from_numpy(lr_aligned_patch / 127.5 - 1.0).unsqueeze(0).float()
        
        return lr_tensor, hr_tensor

# ------------------------------------------------------------
# 損失関数
# ------------------------------------------------------------
class EdgeLoss(torch.nn.Module):
    """画像の勾配マップ（エッジ）間のL1損失を計算するクラス。

    Sobelフィルタを用いてX方向とY方向の勾配を計算し、
    生成画像と教師画像の勾配マップの差の絶対値和の平均を損失とします。
    """
    def __init__(self):
        super().__init__()
        # Sobelフィルタのカーネル (X方向とY方向)
        # requires_grad=False にするため register_buffer を使用
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32).view(1,1,3,3) # Y方向Sobel
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky) # 元のコードではkx.transpose(2,3)だったが、標準的なSobel Yに変更

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """フォワードパス。エッジ損失を計算します。

        Args:
            sr (torch.Tensor): 生成された超解像画像 (B, 1, H, W)。
            hr (torch.Tensor): 対応する高解像度教師画像 (B, 1, H, W)。

        Returns:
            torch.Tensor: 計算されたエッジ損失 (スカラー値)。
        """
        # X方向の勾配
        gx_sr = F.conv2d(sr, self.kx, padding='same') # padding='same'でサイズ維持
        gx_hr = F.conv2d(hr, self.kx, padding='same')
        # Y方向の勾配
        gy_sr = F.conv2d(sr, self.ky, padding='same')
        gy_hr = F.conv2d(hr, self.ky, padding='same')
        
        # 勾配マグニチュードの差ではなく、各方向の勾配差のL1ノルム
        loss_gx = torch.mean(torch.abs(gx_sr - gx_hr))
        loss_gy = torch.mean(torch.abs(gy_sr - gy_hr))
        
        return loss_gx + loss_gy # X方向とY方向の損失の和

class CDLoss(torch.nn.Module):
    """CD (Critical Dimension) 計測に関連する簡易的な損失関数。

    現在の実装では、Y方向の隣接ピクセル間の輝度差（近似的なY方向勾配）の
    絶対値について、生成画像と教師画像の間のL1損失を計算します。
    ラインが主に水平方向に配置されていることを想定しています。
    """
    def __init__(self):
        super().__init__()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """フォワードパス。CD損失を計算します。

        Args:
            sr (torch.Tensor): 生成された超解像画像 (B, 1, H, W)。
            hr (torch.Tensor): 対応する高解像度教師画像 (B, 1, H, W)。

        Returns:
            torch.Tensor: 計算されたCD損失 (スカラー値)。
        """
        # Y方向の輝度差分 (絶対値)
        # sr[:,:,1:,:] は H-1 x W の領域、sr[:,:,:-1,:] も H-1 x W の領域
        diff_y_sr = torch.abs(sr[:,:,1:,:] - sr[:,:,:-1,:])
        diff_y_hr = torch.abs(hr[:,:,1:,:] - hr[:,:,:-1,:])
        
        # 差分のL1損失
        cd_loss = torch.mean(torch.abs(diff_y_sr - diff_y_hr))
        return cd_loss
