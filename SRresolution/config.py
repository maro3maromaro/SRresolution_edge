# ================================================================
# config.py  (共通設定ファイル)  ★軽量モデル対応版★
# ================================================================
"""CD-SEM超解像プロジェクトの共通設定ファイル。

学習データ・検証データのパス、前処理設定、ハイパーパラメータ、
デバイス設定、使用アルゴリズム、モデルパラメータなどを一元管理します。
"""
from pathlib import Path

# ---------------- ディレクトリ設定 ----------------
TRAIN_LOW_DIR: Path | list[Path] = Path('dirA') # 複数の場合は [Path('dirA_1'), Path('dirA_2')]
TRAIN_HIGH_DIR: Path | list[Path] = Path('dirB')

VAL_LOW_DIR: Path | list[Path] = Path('dirC/lr')
VAL_HIGH_DIR: Path | list[Path] = Path('dirC/hr')

# ---------------- 前処理設定 ----------------
FLAT_PATH: Path | None = None
DARK_PATH: Path | None = None
FILE_PREFIX: str = 'ABC'
# ---------------- 画像処理設定 ----------------
# 学習/検証時に使用する画像の中央領域のサイズ
CENTRAL_CROP_HEIGHT: int = 800
"""画像から切り出す中央領域の高さ（ピクセル）。"""
CENTRAL_CROP_WIDTH: int = 256
"""画像から切り出す中央領域の幅（ピクセル）。"""

# ---------------- ハイパーパラメータ ----------------
EPOCHS: int = 50
BATCH_SIZE: int = 8
# PATCH_SIZE は CENTRAL_CROP_WIDTH と CENTRAL_CROP_HEIGHT 以下である必要があります。
# 特に、正方形パッチを想定しているため、min(CENTRAL_CROP_HEIGHT, CENTRAL_CROP_WIDTH) 以下が安全です。
# 例: CENTRAL_CROP_WIDTH が 256 の場合、PATCH_SIZE は 256 以下。
PATCH_SIZE: int = 128 # GPUメモリに応じて調整 (例: 256 -> 128)
LEARNING_RATE: float = 2e-4
NUM_WORKERS: int = 2 # 環境に応じて調整 (例: 4 -> 2)
SEED: int = 42

# ---------------- 損失関数の重み ----------------
LOSS_WEIGHTS: dict[str, float] = {
    'l1': 1.0,
    'ssim': 0.1,
    'edge': 0.05,
    'cd': 0.5
}

# ---------------- 学習率スケジューラ設定 ----------------
LR_SCHEDULER_TYPE: str = 'cosine'
LR_SCHEDULER_PARAMS: dict = {
    'cosine': {'T_max': EPOCHS, 'eta_min': LEARNING_RATE * 0.01},
    'step': {'step_size': max(1, EPOCHS // 2), 'gamma': 0.5} # EPOCHSに応じて調整
}

# ---------------- モデルチェックポインティング設定 ----------------
MONITOR_METRIC: str = 'val_cdloss'
MONITOR_MODE: str = 'min'

# ---------------- デバイス設定 ----------------
DEVICE: str = 'cuda' # 'cuda' or 'cpu'

# ---------------- モデルアーキテクチャとパラメータ設定 ----------------
# 各モデルのパラメータ (主に軽量化のため)
# num_feat: 特徴マップのチャネル数 (EDSR, RDN, SRMDなど) / nf: フォールバック実装のチャネル数
# num_block: 主要な残差ブロックやDenseブロックの数 / nb: フォールバック実装のブロック数
MODEL_CONFIGS: dict[str, dict] = {
    'edgeformer_edsr': {'num_feat': 64, 'num_block': 16, 'edgeformer_dim': 64, 'upscale': 1},
    'edgeformer_edsr_light': {'num_feat': 32, 'num_block': 8, 'edgeformer_dim': 32, 'upscale': 1},
    
    'rdn_edge': {'num_feat': 64, 'num_block': 4, 'num_dense_block': 6, 'growth_rate': 32, 'upscale': 1}, # num_blockはRDBの数
    'rdn_edge_light': {'num_feat': 32, 'num_block': 3, 'num_dense_block': 4, 'growth_rate': 16, 'upscale': 1},
    
    'srmd': {'num_feat': 64, 'num_block': 12, 'upscale': 1},
    'srmd_light': {'num_feat': 32, 'num_block': 6, 'upscale': 1},
    
    'swinir': { # SwinIRは元々パラメータが多いので "light" は相対的に
        'upscale': 1, 'img_size': PATCH_SIZE, 'window_size': 8, 'img_range': 1.,
        'embed_dim': 60, 'depths': [6, 6, 6, 6], 'num_heads': [6, 6, 6, 6], 'mlp_ratio': 2, 'upsampler': ''
    },
    'swinir_tiny': { # さらに軽量化を試みる場合
        'upscale': 1, 'img_size': PATCH_SIZE, 'window_size': 8, 'img_range': 1.,
        'embed_dim': 48, 'depths': [4, 4, 4, 4], 'num_heads': [4, 4, 4, 4], 'mlp_ratio': 2, 'upsampler': ''
    },

    'nafnet': {'img_channel': 1, 'width': 32, 'middle_blk_num': 1, 'enc_blk_nums': [1, 1, 1, 2], 'dec_blk_nums': [1, 1, 1, 1], 'upscale': 1},
    'nafnet_light': {'img_channel': 1, 'width': 16, 'middle_blk_num': 1, 'enc_blk_nums': [1, 1, 1, 1], 'dec_blk_nums': [1, 1, 1, 1], 'upscale': 1},
    
    'simplesrnet': {'nf': 32, 'nb': 3, 'upscale': 1} # 新しい超軽量モデル
}

ALGO: str = 'simplesrnet' # デフォルトを最も軽いものの一つに変更

ALGO_LIST: list[str] = [
    'edgeformer_edsr', 'edgeformer_edsr_light',
    'rdn_edge', 'rdn_edge_light',
    'srmd', 'srmd_light',
    'swinir', 'swinir_tiny',
    'nafnet', 'nafnet_light',
    'simplesrnet'
]
"""利用可能な超解像アルゴリズムのリスト。"""
