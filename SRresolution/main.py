# ================================================================
# main.py  (エントリーポイント: 学習 + 評価 ワンコマンド実行)
# ================================================================
#  使い方:
#     $ python main.py                # config.py に従って学習 + 検証
#     $ python main.py --algo srmd    # アルゴリズムのみ上書き
# ------------------------------------------------
#  動作概要:
#     1. config.py からパス & パラメータ読込
#     2. 学習用 (dirA/dirB) と検証用 (dirC) を DataLoader 化
#     3. Trainer で所定エポック学習
#     4. 学習後に検証セットで PSNR / SSIM / CDLoss を計算
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
    p = argparse.ArgumentParser(description='CD‑SEM 超解像：学習 + 検証')
    p.add_argument('--algo', default=config.ALGO, choices=config.ALGO_LIST,
                   help='使用アルゴリズム (config.py で一覧定義)')
    return p.parse_args()

# ------------------------------------------------------------
# 検証関数
# ------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: DataLoader, device: str):
    model.eval()
    cd_metric = CDLoss().to(device)
    psnr_sum = ssim_sum = cd_sum = 0.0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr_sum += psnr(sr, hr, data_range=2.).item()
            ssim_sum += ssim(sr, hr, data_range=2., reduction='elementwise_mean').item()
            cd_sum   += cd_metric(sr, hr).item()
    n = len(loader)
    return psnr_sum/n, ssim_sum/n, cd_sum/n

# ------------------------------------------------------------
# メイン関数
# ------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(42)

    # ---------------- Dataset / DataLoader ----------------
    train_ds = SEMPatchDataset(low_dir=config.TRAIN_LOW_DIR,
                               high_dir=config.TRAIN_HIGH_DIR,
                               flat_path=config.FLAT_PATH,
                               dark_path=None,
                               patch=config.PATCH_SIZE,
                               stride=config.STRIDE)

    val_ds   = SEMPatchDataset(low_dir=config.VAL_LOW_DIR,
                               high_dir=config.VAL_HIGH_DIR,
                               flat_path=config.FLAT_PATH,
                               dark_path=None,
                               patch=config.PATCH_SIZE,
                               stride=config.STRIDE)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ---------------- モデル & Trainer ----------------
    model = build_model(args.algo).to(config.DEVICE)
    save_root = Path('runs') / args.algo
    save_root.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, train_loader, device=config.DEVICE,
                      save_dir=save_root, epochs=config.EPOCHS,
                      lr=config.LEARNING_RATE)

    # ---------------- 学習 ----------------
    print('==> Training start')
    trainer.train()

    # ---------------- 検証 ----------------
    print('==> Evaluation start')
    p,s,c = evaluate(model, val_loader, config.DEVICE)
    print(f'Validation PSNR  : {p:.2f} dB')
    print(f'Validation SSIM  : {s:.4f}')
    print(f'Validation CDLoss: {c:.6f}')

if __name__ == '__main__':
    main()
