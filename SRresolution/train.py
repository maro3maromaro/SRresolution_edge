# ================================================================
# train.py  (学習ループ管理モジュール)
# ================================================================
#  Trainer クラスは CD‑SEM 超解像アルゴリズムの学習を一括で管理します。
#  主な役割：
#    1. DataLoader からバッチを取得しモデルへ入力
#    2. 総合損失 (L1 + SSIM + EdgeLoss + CDLoss) を計算
#    3. Mixed‑Precision (fp16) によるバックプロパゲーション
#    4. 一定ステップごとに学習ログを出力
#    5. エポック終了後にモデル重みを保存
# ---------------------------------------------------------------
#  ※ 入力・出力とも 1 チャネル画像、upscale=1 を想定しています。
# ================================================================

from pathlib import Path
import torch
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

from utils import EdgeLoss, CDLoss

class Trainer:
    """超解像モデルを学習させる高レベル管理クラス"""

    # ------------------------------------------------------------
    # コンストラクタ
    # ------------------------------------------------------------
    def __init__(self,
                 model: torch.nn.Module,
                 loader: torch.utils.data.DataLoader,
                 device: str,
                 save_dir: Path,
                 epochs: int = 50,
                 lr: float = 2e-4):
        # ----- オブジェクト保持 -----
        self.model    = model
        self.loader   = loader
        self.device   = device
        self.save_dir = Path(save_dir)
        self.epochs   = epochs

        # ----- Optimizer & Scheduler -----
        # AdamW は weight decay を取り入れた Adam。β 値は論文推奨値に近い (0.9,0.99)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        # ----- 損失関数 -----
        # L1: 画素誤差 / SSIM: 構造類似度 / EdgeLoss: エッジ形状 / CDLoss: ライン幅
        self.l1        = nn.L1Loss()
        self.ssim      = StructuralSimilarityIndexMeasure(data_range=2.0)  # 入力正規化 [-1,1] → range=2
        self.edge_loss = EdgeLoss()
        self.cd_loss   = CDLoss()

        # ----- Mixed Precision 用 GradScaler -----
        self.scaler = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------
    # _loss(): 各損失を合成して最終スカラーを返す
    # ------------------------------------------------------------
    def _loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """総合損失 L_total を計算"""
        l1   = self.l1(sr, hr)
        ssim = 1.0 - self.ssim(sr, hr)  # SSIM は高いほど良い → 1-SSIM で損失化
        edge = self.edge_loss(sr, hr)
        cd   = self.cd_loss(sr, hr)
        # 重みは CD 精度を重視した経験値
        return l1 + 0.1*ssim + 0.05*edge + 0.5*cd

    # ------------------------------------------------------------
    # train(): 指定エポック数だけトレーニングを実行
    # ------------------------------------------------------------
    def train(self):
        self.model.train()  # バッチ正規化・Dropout を学習モードに

        for epoch in range(1, self.epochs + 1):
            for step, (lr_img, hr_img) in enumerate(self.loader, start=1):
                # ---- デバイス転送 ----
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                # ---- forward & loss (FP16) ----
                # autocast() コンテキストで推論・損失計算を fp16 に省メモリ化
                with torch.cuda.amp.autocast():
                    sr   = self.model(lr_img)      # Super‑resolved 画像
                    loss = self._loss(sr, hr_img)

                # ---- backward ----
                self.optim.zero_grad()             # 勾配初期化
                self.scaler.scale(loss).backward() # 勾配をスケールして backward
                self.scaler.step(self.optim)       # Optimizer step (fp32)
                self.scaler.update()               # スケーラのスケール値更新

                # ---- 進捗ログ ----
                if step % 100 == 0:
                    print(f"Epoch {epoch:03d}  Step {step:05d}  Loss {loss.item():.4f}")

            # ========== エポック終了：モデル重み保存 ==========
            save_path = self.save_dir / f'epoch_{epoch:03d}.pth'
            torch.save(self.model.state_dict(), save_path)
            print(f"[Save] → {save_path}")
