# ================================================================
# train.py  (学習ループ管理モジュール) ★2025‑05 改善提案版★
# ================================================================
"""超解像モデルの学習と検証を行うTrainerクラスを定義するモジュール。

主な機能:
  - DataLoaderからバッチを取得しモデルへ入力
  - 複数の損失関数（L1, SSIM, EdgeLoss, CDLoss）を組み合わせた総合損失の計算
  - Mixed-Precision (fp16) を利用したバックプロパゲーション
  - 学習の進捗ログ出力
  - エポックごとの検証と、指定されたメトリックに基づくベストモデルの保存
  - 学習率スケジューラの適用
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional import peak_signal_noise_ratio as psnr_functional

import numpy as np
from pathlib import Path
import config # 設定ファイルをインポート
from utils import EdgeLoss, CDLoss # utils.py から損失関数をインポート

class Trainer:
    """超解像モデルを学習・検証させる高レベル管理クラス。

    Attributes:
        model (nn.Module): 学習対象の超解像モデル。
        train_loader (DataLoader): 学習用データのDataLoader。
        val_loader (DataLoader): 検証用データのDataLoader。
        device (str): 学習に使用するデバイス ('cuda' or 'cpu')。
        save_dir (Path): モデルやログの保存先ディレクトリ。
        epochs (int): 総学習エポック数。
        lr (float): 初期学習率。
        optim (torch.optim.Optimizer): オプティマイザ。
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): 学習率スケジューラ。
        l1_loss_fn (nn.L1Loss): L1損失関数。
        ssim_metric (StructuralSimilarityIndexMeasure): SSIM計算用メトリック。
        edge_loss_fn (EdgeLoss): エッジ損失関数。
        cd_loss_fn (CDLoss): CD損失関数。
        loss_weights (dict[str, float]): 各損失の重み。
        scaler (torch.cuda.amp.GradScaler): Mixed Precision用GradScaler。
        monitor_metric (str): ベストモデル保存時に監視するメトリック名。
        monitor_mode (str): 監視メトリックのモード ('min' or 'max')。
        best_metric_val (float): 保存されているベストメトリックの値。
        best_model_path (Path): ベストモデルの保存パス。
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str,
                 save_dir: Path,
                 epochs: int = config.EPOCHS,
                 lr: float = config.LEARNING_RATE):
        """
        Args:
            model (nn.Module): 学習対象のモデル。
            train_loader (DataLoader): 学習用データローダー。
            val_loader (DataLoader): 検証用データローダー。
            device (str): 使用デバイス ('cuda' or 'cpu')。
            save_dir (Path): モデルやログの保存先ディレクトリ。
            epochs (int, optional): 学習エポック数。config.EPOCHSがデフォルト。
            lr (float, optional): 初期学習率。config.LEARNING_RATEがデフォルト。
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.epochs = epochs
        self.lr = lr

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99))

        self.scheduler = None
        if config.LR_SCHEDULER_TYPE == 'cosine':
            params = config.LR_SCHEDULER_PARAMS.get('cosine', {'T_max': epochs, 'eta_min': lr * 0.01})
            self.scheduler = CosineAnnealingLR(self.optim, **params)
        elif config.LR_SCHEDULER_TYPE == 'step':
            params = config.LR_SCHEDULER_PARAMS.get('step', {'step_size': 30, 'gamma': 0.5})
            self.scheduler = StepLR(self.optim, **params)

        self.l1_loss_fn = nn.L1Loss().to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)
        self.edge_loss_fn = EdgeLoss().to(self.device)
        self.cd_loss_fn = CDLoss().to(self.device)

        self.loss_weights = config.LOSS_WEIGHTS

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        self.monitor_metric = config.MONITOR_METRIC
        self.monitor_mode = config.MONITOR_MODE
        self.best_metric_val = float('inf') if self.monitor_mode == 'min' else float('-inf')
        self.best_model_path = self.save_dir / f'best_model_on_{self.monitor_metric}.pth'


    def _calculate_combined_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """重み付けされた総合損失 (L_total) と各損失成分を計算します。

        Args:
            sr (torch.Tensor): モデルが出力した超解像画像。
            hr (torch.Tensor): 対応する高解像度教師画像。

        Returns:
            tuple[torch.Tensor, dict[str, float]]:
                - total_loss (torch.Tensor): 計算された総合損失。
                - loss_dict (dict[str, float]): 各損失成分の値を持つ辞書
                  ('l1', 'ssim_loss', 'edge', 'cd')。
        """
        loss_l1 = self.l1_loss_fn(sr, hr)
        # ssim_metric はメトリックそのものを返す (高いほど良い)
        # 損失として使う場合は 1.0 から引く
        current_ssim = self.ssim_metric(sr, hr)
        loss_ssim_val = 1.0 - current_ssim # 損失としてのSSIM (低いほど良い)
        
        loss_edge = self.edge_loss_fn(sr, hr)
        loss_cd = self.cd_loss_fn(sr, hr)

        total_loss = (self.loss_weights['l1'] * loss_l1 +
                      self.loss_weights['ssim'] * loss_ssim_val +
                      self.loss_weights['edge'] * loss_edge +
                      self.loss_weights['cd'] * loss_cd)
        
        loss_dict = {
            "l1": loss_l1.item(),
            "ssim_metric": current_ssim.item(), # メトリックとしてのSSIM値
            "ssim_loss": loss_ssim_val.item(),   # (1-SSIM) の損失値
            "edge": loss_edge.item(),
            "cd": loss_cd.item(),
            "total": total_loss.item() # For logging convenience
        }
        return total_loss, loss_dict

    def _train_one_epoch(self, epoch: int) -> float:
        """1エポック分の学習処理を実行します。

        Args:
            epoch (int): 現在のエポック番号。

        Returns:
            float: このエポックでの平均学習損失。
        """
        self.model.train()
        total_train_loss = 0.0
        
        for step, (lr_img, hr_img) in enumerate(self.train_loader, start=1):
            lr_img = lr_img.to(self.device)
            hr_img = hr_img.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                sr = self.model(lr_img)
                loss, loss_components = self._calculate_combined_loss(sr, hr_img)

            self.optim.zero_grad()
            if self.device == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                self.optim.step()
            
            total_train_loss += loss.item()

            if step % 100 == 0:
                log_msg = (f"Epoch {epoch:03d} Step {step:05d}/{len(self.train_loader)} "
                           f"Train Loss: {loss.item():.4f} ("
                           f"L1: {loss_components['l1']:.4f}, "
                           f"SSIM: {loss_components['ssim_metric']:.4f}, " # ssim_metricを表示
                           f"Edge: {loss_components['edge']:.4f}, "
                           f"CD: {loss_components['cd']:.4f})")
                print(log_msg)
        
        avg_train_loss = total_train_loss / len(self.train_loader)
        print(f"Epoch {epoch:03d} Average Train Loss: {avg_train_loss:.4f}")
        return avg_train_loss

    def _validate_one_epoch(self, epoch: int) -> dict[str, float]:
        """1エポック分の検証処理を実行します。

        Args:
            epoch (int): 現在のエポック番号。

        Returns:
            dict[str, float]: 検証結果のメトリックを含む辞書
                ('val_loss', 'val_psnr', 'val_ssim', 'val_cdloss')。
        """
        self.model.eval()
        total_val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0 # メトリックとしてのSSIM
        val_cdloss_sum = 0.0
        
        with torch.no_grad():
            for lr_img, hr_img in self.val_loader:
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    sr = self.model(lr_img)
                    loss, _ = self._calculate_combined_loss(sr, hr_img)
                
                total_val_loss += loss.item()
                val_psnr_sum += psnr_functional(sr, hr_img, data_range=2.0).item()
                val_ssim_sum += self.ssim_metric(sr, hr_img).item()
                val_cdloss_sum += self.cd_loss_fn(sr, hr_img).item() # CDLossは低いほど良い

        num_val_batches = len(self.val_loader)
        if num_val_batches == 0:
            print(f"Epoch {epoch:03d} Validation loader is empty. Skipping validation.")
            return {"val_loss": float('inf'), "val_psnr": 0.0, "val_ssim": 0.0, "val_cdloss": float('inf')}

        avg_val_loss = total_val_loss / num_val_batches
        avg_val_psnr = val_psnr_sum / num_val_batches
        avg_val_ssim = val_ssim_sum / num_val_batches
        avg_val_cdloss = val_cdloss_sum / num_val_batches

        print(f"Epoch {epoch:03d} Validation Avg Loss: {avg_val_loss:.4f}, "
              f"PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}, CDLoss: {avg_val_cdloss:.6f}")
        
        return {
            "val_loss": avg_val_loss,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim, # SSIMメトリック (高いほど良い)
            "val_cdloss": avg_val_cdloss # CDLoss (低いほど良い)
        }

    def train(self):
        """指定されたエポック数だけモデルの学習と検証を実行します。

        エポックごとに検証を行い、`monitor_metric`に基づいて
        最も性能の良いモデルの重みを保存します。
        """
        print(f"Training started on {self.device} for {self.epochs} epochs.")
        print(f"Saving models to: {self.save_dir}")
        print(f"Monitoring metric: {self.monitor_metric} (mode: {self.monitor_mode}) for best model.")

        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)
            val_metrics = self._validate_one_epoch(epoch)

            if self.scheduler:
                # ReduceLROnPlateauのようなスケジューラはメトリックを渡す必要がある
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.monitor_metric, float('inf')))
                else:
                    self.scheduler.step()
                print(f"Epoch {epoch:03d} Current LR: {self.optim.param_groups[0]['lr']:.6e}")

            current_metric_val = val_metrics.get(self.monitor_metric)
            if current_metric_val is not None:
                save_checkpoint = False
                if self.monitor_mode == 'min' and current_metric_val < self.best_metric_val:
                    self.best_metric_val = current_metric_val
                    save_checkpoint = True
                elif self.monitor_mode == 'max' and current_metric_val > self.best_metric_val:
                    self.best_metric_val = current_metric_val
                    save_checkpoint = True
                
                if save_checkpoint:
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f"Epoch {epoch:03d} New best model saved with {self.monitor_metric}: {self.best_metric_val:.6f} to {self.best_model_path}")

            # オプション: 各エポックのモデルも保存する場合
            # epoch_save_path = self.save_dir / f'epoch_{epoch:03d}.pth'
            # torch.save(self.model.state_dict(), epoch_save_path)
            # print(f"Epoch {epoch:03d} Model for epoch saved to {epoch_save_path}")

        print(f"Training finished. Best model ({self.monitor_metric}: {self.best_metric_val:.6f}) saved at {self.best_model_path}")
