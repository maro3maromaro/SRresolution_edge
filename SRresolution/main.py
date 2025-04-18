# ================================================================
# main.py  (train + eval, flat/dark 補正なし, Dataset 引数修正)
# ================================================================
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

import config
from utils import SEMPatchDataset, seed_everything, CDLoss
from train import Trainer
from model import build_model

# ------------------------------------------------------------
# 引数パーサ
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='CD‑SEM 超解像: Train + Eval')
    p.add_argument('--algo', default=config.ALGO, choices=config.ALGO_LIST,
                   help='使用アルゴリズム')
    return p.parse_args()

# ------------------------------------------------------------
# 評価関数
# ------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader, device: str):
    model.eval()
    cd_metric = CDLoss().to(device)
    p_sum = s_sum = c_sum = 0.0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            p_sum += psnr(sr, hr, data_range=2.).item()
            s_sum += ssim(sr, hr, data_range=2., reduction='elementwise_mean').item()
            c_sum += cd_metric(sr, hr).item()
    n = len(loader)
    return p_sum / n, s_sum / n, c_sum / n

# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(42)

    # ----- DataLoader -----
    train_ds = SEMPatchDataset(low_dir=config.TRAIN_LOW_DIRS,
                               high_dir=config.TRAIN_HIGH_DIRS,
                               patch=config.PATCH_SIZE,
                               stride=config.STRIDE)

    val_ds   = SEMPatchDataset(low_dir=config.VAL_LOW_DIRS,
                               high_dir=config.VAL_HIGH_DIRS,
                               patch=config.PATCH_SIZE,
                               stride=config.STRIDE)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ----- Model & Trainer -----
    model = build_model(args.algo).to(config.DEVICE)
    save_root = Path('runs') / args.algo
    save_root.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, train_loader, device=config.DEVICE,
                      save_dir=save_root, epochs=config.EPOCHS,
                      lr=config.LEARNING_RATE)

    print('==> Training start')
    trainer.train()

    print('==> Evaluation start')
    p, s, c = evaluate(model, val_loader, config.DEVICE)
    print(f'Validation PSNR  : {p:.2f} dB')
    print(f'Validation SSIM  : {s:.4f}')
    print(f'Validation CDLoss: {c:.6f}')

if __name__ == '__main__':
    main()
