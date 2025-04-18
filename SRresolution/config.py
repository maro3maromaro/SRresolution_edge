# ================================================================
# config.py  (共通設定ファイル)  ★2025‑04 修正版★
# ================================================================
from pathlib import Path

# ---------------- ディレクトリ設定 ----------------
# 低画質 (LR) 学習データ : dirA
TRAIN_LOW_DIR  = Path('dirA')
# 高画質 (HR) 学習データ : dirB
TRAIN_HIGH_DIR = Path('dirB')

# 検証用ペア (1024 枚)
VAL_LOW_DIR    = Path('dirC/lr')   # 低画質検証
VAL_HIGH_DIR   = Path('dirC/hr')   # 高画質検証

# ---------------- 前処理設定 ----------------
FLAT_PATH      = None   # フラット補正無し
DARK_PATH      = None   # ダーク補正無し
FILE_PREFIX    = 'ABC'  # 画像ファイル名が "ABC***.jpg" のみ対象

# ---------------- ハイパーパラメータ ----------------
EPOCHS         = 50
BATCH_SIZE     = 16
PATCH_SIZE     = 256
STRIDE         = 64
LEARNING_RATE  = 2e-4

# ----------------デバイス / アルゴリズム ----------------
DEVICE         = 'cuda'
ALGO           = 'edgeformer_edsr'
ALGO_LIST      = ['rdn_edge','edgeformer_edsr','srmd','swinir_light','nafnet']