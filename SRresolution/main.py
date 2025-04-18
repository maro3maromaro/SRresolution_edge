# ================================================================
# train.py  (学習ループ)
# ================================================================
#  Trainer クラスは以下の役割：
#    1. バッチ単位で forward → 損失を計算
#    2. Mixed‑Precision (fp16) で backward / optimizer step
#    3. 100 ステップおきに進捗をプリント
#    4. 各エポック終了時にモデル重みを保存
# ------------------------------------------------
#  総合損失 L_total = L1 + 0.1*(1‑SSIM) + 0.05*EdgeLoss + 0.5*CDLoss
# ------------------------------------------------

from pathlib import Path
import torch
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

from utils import EdgeLoss, CDLoss

class Trainer:
    """超解像モデルの学習管理クラス"""

    def __init__(self, model, loader, device: str, save_dir: Path,
                 epochs: int = 50, lr: float = 2e-4):
        # ===== オブジェクト保持 =====
        self.model    = model
        self.loader   = loader
        self.device   = device
        self.save_dir = Path(save_dir)
        self.epochs   = epochs

        # --- Optimizer (AdamW) ---
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        # --- 損失関数セット ---
        self.l1        = nn.L1Loss()                                    # 画素 L1
        self.ssim      = StructuralSimilarityIndexMeasure(data_range=2.)# SSIM
        self.edge_loss = EdgeLoss()                                     # Canny/Sobel Edge
        self.cd_loss   = CDLoss()                                       # CD 寸法誤差

        # --- fp16 スケーラ ---
        self.scaler = torch.cuda.amp.GradScaler()

    # ------------------------------------------------------------
    # 内部：総合損失計算
    # ------------------------------------------------------------
    def _loss(self, sr, hr):
        l1   = self.l1(sr, hr)
        ssim = 1.0 - self.ssim(sr, hr)
        edge = self.edge_loss(sr, hr)
        cd   = self.cd_loss(sr, hr)
        return l1 + 0.1*ssim + 0.05*edge + 0.5*cd

    # ------------------------------------------------------------
    # 公開：学習開始
    # ------------------------------------------------------------
    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for step, (lr_img, hr_img) in enumerate(self.loader, start=1):
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

                # ---- forward & loss (fp16) ----
                with torch.cuda.amp.autocast():
                    sr   = self.model(lr_img)
                    loss = self._loss(sr, hr_img)

                # ---- backward ----
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                if step % 100 == 0:
                    print(f"Epoch {epoch:03d} Step {step:05d} Loss {loss.item():.4f}")

            # === save ===
            save_path = self.save_dir / f'epoch_{epoch:03d}.pth'
            torch.save(self.model.state_dict(), save_path)
            print(f"[Save] → {save_path}")
