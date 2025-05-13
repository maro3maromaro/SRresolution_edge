# ================================================================
# config.py  (共通設定ファイル)  ★2025‑04 修正版★
# ================================================================
"""CD-SEM超解像プロジェクトの共通設定ファイル。

学習データ・検証データのパス、前処理設定、ハイパーパラメータ、
デバイス設定、使用アルゴリズムなどを一元管理します。
"""
from pathlib import Path

# ---------------- ディレクトリ設定 ----------------
# 低画質 (LR) 学習データ : dirA
TRAIN_LOW_DIR: Path = Path('dirA')
"""低画質画像の学習データが格納されているディレクトリのパス。"""

# 高画質 (HR) 学習データ : dirB
TRAIN_HIGH_DIR: Path = Path('dirB')
"""高画質画像の学習データが格納されているディレクトリのパス。"""

# 検証用ペア (1024 枚)
VAL_LOW_DIR: Path = Path('dirC/lr')   # 低画質検証
"""低画質画像の検証データが格納されているディレクトリのパス。"""

VAL_HIGH_DIR: Path = Path('dirC/hr')   # 高画質検証
"""高画質画像の検証データが格納されているディレクトリのパス。"""

# ---------------- 前処理設定 ----------------
FLAT_PATH: Path | None = None   # フラット補正無し
"""フラット補正に使用する画像のパス。Noneの場合は補正なし。"""

DARK_PATH: Path | None = None   # ダーク補正無し
"""ダーク補正に使用する画像のパス。Noneの場合は補正なし。"""

FILE_PREFIX: str = 'ABC'  # 画像ファイル名が "ABC***.jpg" のみ対象
"""処理対象とする画像ファイルの接頭辞。"""

# ---------------- ハイパーパラメータ ----------------
EPOCHS: int = 50
"""学習のエポック数。"""

BATCH_SIZE: int = 16
"""学習および検証時のバッチサイズ。"""

PATCH_SIZE: int = 256
"""学習時に画像から切り出すパッチのサイズ（正方形を想定）。"""

STRIDE: int = 64
"""パッチを切り出す際のストライド。utils.SEMPatchDatasetの実装によっては未使用。"""

LEARNING_RATE: float = 2e-4
"""オプティマイザの学習率。"""

# ----------------デバイス / アルゴリズム ----------------
DEVICE: str = 'cuda'
"""学習および推論に使用するデバイス ('cuda' または 'cpu')。"""

ALGO: str = 'edgeformer_edsr'
"""デフォルトで使用する超解像アルゴリズムの名前。"""

ALGO_LIST: list[str] = ['rdn_edge','edgeformer_edsr','srmd','swinir_light','nafnet']
"""利用可能な超解像アルゴリズムのリスト。"""

# ---------------- 学習率スケジューラ設定 (train.py 改善案で追加された想定) ----------------
LR_SCHEDULER_TYPE: str = 'cosine'
"""学習率スケジューラの種類 ('cosine', 'step', 'none' など)。"""

LR_SCHEDULER_PARAMS: dict = {
    'cosine': {'T_max': EPOCHS, 'eta_min': LEARNING_RATE * 0.01},
    'step': {'step_size': 30, 'gamma': 0.5}
}
"""学習率スケジューラごとのパラメータ。"""

# ---------------- モデルチェックポインティング設定 (train.py 改善案で追加された想定) ----------------
MONITOR_METRIC: str = 'val_cdloss'
"""モデル保存の際に監視する評価指標 ('val_loss', 'val_psnr', 'val_ssim', 'val_cdloss' など)。"""

MONITOR_MODE: str = 'min'
"""監視する評価指標のモード ('min' なら小さいほど良い、'max' なら大きいほど良い)。"""

# ---------------- その他 (train.py 改善案で追加された想定) ----------------
NUM_WORKERS: int = 4
"""DataLoaderで使用するワーカープロセス数。"""

SEED: int = 42
"""各種乱数生成器のシード値。"""

# ---------------- 損失関数の重み (train.py 改善案で追加された想定) ----------------
LOSS_WEIGHTS: dict[str, float] = {
    'l1': 1.0,
    'ssim': 0.1,
    'edge': 0.05,
    'cd': 0.5
}
"""各損失関数の重み。L_total = w_l1*L1 + w_ssim*(1-SSIM) + w_edge*EdgeLoss + w_cd*CDLoss のように使用。"""
