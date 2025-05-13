# ================================================================
# main.py  (train + eval, flat/dark 補正なし, Dataset 引数修正)
# ================================================================
"""CD-SEM超解像モデルの学習と評価を実行するメインスクリプト。

コマンドライン引数から使用するアルゴリズムを指定し、
設定ファイル (config.py) に基づいて学習と評価を行います。
学習後には、検証データセットに対する最終評価も実行します。
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

import config
from utils import SEMPatchDataset, seed_everything, CDLoss
from train import Trainer # train.py の Trainer クラス (改善案を想定)
from model import build_model

# ------------------------------------------------------------
# 引数パーサ
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します。

    Returns:
        argparse.Namespace: 解析されたコマンドライン引数。
            'algo' (str): 使用する超解像アルゴリズム名。
    """
    p = argparse.ArgumentParser(description='CD‑SEM 超解像: Train + Eval')
    p.add_argument('--algo', default=config.ALGO, choices=config.ALGO_LIST,
                   help='使用アルゴリズム')
    # config.py で設定されている他のパラメータも引数で上書き可能にする場合はここに追加
    # 例: p.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of training epochs')
    return p.parse_args()

# ------------------------------------------------------------
# 評価関数 (Trainerに検証機能が移ったため、最終評価専用に)
# ------------------------------------------------------------

def evaluate_final_model(model_path: Path, algo: str, loader: DataLoader, device: str) -> tuple[float, float, float]:
    """指定されたモデルファイルを使用して、検証データセットで最終評価を行います。

    Args:
        model_path (Path): 評価する学習済みモデルのファイルパス。
        algo (str): モデル構築に使用したアルゴリズム名。
        loader (DataLoader): 検証用データのDataLoader。
        device (str): 評価に使用するデバイス ('cuda' or 'cpu')。

    Returns:
        tuple[float, float, float]: (平均PSNR, 平均SSIM, 平均CDLoss) のタプル。
    """
    print(f"\nEvaluating model: {model_path}")
    model = build_model(algo).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return 0.0, 0.0, 0.0
    model.eval()

    cd_metric_fn = CDLoss().to(device)
    # torchmetricsのメトリックインスタンス化 (SSIM)
    # train.pyの改善案に合わせて、SSIMもtorchmetrics.imageから取得
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    # torchmetrics.functional (PSNR)
    from torchmetrics.functional import peak_signal_noise_ratio as psnr_functional

    p_sum = s_sum = c_sum = 0.0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            p_sum += psnr_functional(sr, hr, data_range=2.0).item()
            s_sum += ssim_metric_fn(sr, hr).item()
            c_sum += cd_metric_fn(sr, hr).item()
            
    n = len(loader)
    if n == 0:
        print("Warning: Validation loader is empty. Cannot evaluate.")
        return 0.0, 0.0, 0.0
        
    avg_psnr = p_sum / n
    avg_ssim = s_sum / n
    avg_cdloss = c_sum / n
    
    print(f'Final Validation PSNR  : {avg_psnr:.2f} dB')
    print(f'Final Validation SSIM  : {avg_ssim:.4f}')
    print(f'Final Validation CDLoss: {avg_cdloss:.6f}')
    return avg_psnr, avg_ssim, avg_cdloss

# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

def main():
    """メイン実行関数。

    コマンドライン引数を解析し、データローダーを準備し、
    指定されたアルゴリズムでモデルの学習と評価を行います。
    """
    args = parse_args()
    seed_everything(config.SEED if hasattr(config, 'SEED') else 42) # configにSEEDがあれば使う

    # ----- DataLoader -----
    # config.py の TRAIN_LOW_DIR などが単一PathかリストPathかに対応
    # (config.pyの改善案を反映)
    def _ensure_list_of_paths(dir_config: Path | list[Path]) -> list[Path]:
        if isinstance(dir_config, Path):
            return [dir_config]
        elif isinstance(dir_config, list) and all(isinstance(p, Path) for p in dir_config):
            return dir_config
        elif dir_config is None: # ディレクトリが指定されていない場合も考慮
            return []
        else:
            raise ValueError(f"Directory configuration must be a Path or list of Paths, got {type(dir_config)}")

    train_low_dirs = _ensure_list_of_paths(config.TRAIN_LOW_DIR)
    train_high_dirs = _ensure_list_of_paths(config.TRAIN_HIGH_DIR)
    val_low_dirs = _ensure_list_of_paths(config.VAL_LOW_DIR)
    val_high_dirs = _ensure_list_of_paths(config.VAL_HIGH_DIR)

    if not train_low_dirs or not train_high_dirs:
        print("Training directories are not properly configured. Exiting.")
        return
    if not val_low_dirs or not val_high_dirs:
        print("Validation directories are not properly configured. Evaluation might fail.")
        # 検証データなしで学習だけ行う場合は、以下の val_ds, val_loader の処理をスキップするなどの対応が必要

    train_ds = SEMPatchDataset(low_dir=train_low_dirs,
                               high_dir=train_high_dirs,
                               patch=config.PATCH_SIZE,
                               # stride=config.STRIDE, # SEMPatchDatasetの実装による
                               repeats=4) # repeatsはDataset内で定義

    val_ds = SEMPatchDataset(low_dir=val_low_dirs,
                             high_dir=val_high_dirs,
                             patch=config.PATCH_SIZE,
                             # stride=config.STRIDE,
                             repeats=1) # 検証時は通常repeats=1

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, 
                              num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4, 
                              pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, 
                            num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4, 
                            pin_memory=True)

    # ----- Model & Trainer -----
    model = build_model(args.algo) # デバイスへの .to(config.DEVICE) は Trainer 内で行う
    
    save_root = Path('runs') / args.algo
    save_root.mkdir(parents=True, exist_ok=True)

    # Trainerの初期化 (val_loader を渡す)
    # (config.pyとtrain.pyの改善案を反映)
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=config.DEVICE,
                      save_dir=save_root,
                      epochs=config.EPOCHS,
                      lr=config.LEARNING_RATE)

    print(f'==> Training start for algorithm: {args.algo}')
    trainer.train() # 学習実行 (内部で検証とベストモデル保存も行う)

    print('\n==> Final Evaluation on Validation Set using the best saved model')
    # 学習中に保存されたベストモデルをロードして評価
    best_model_path = trainer.best_model_path if hasattr(trainer, 'best_model_path') else save_root / f'best_model_{config.MONITOR_METRIC if hasattr(config, "MONITOR_METRIC") else "unknown"}.pth'
    
    if best_model_path.exists():
        evaluate_final_model(model_path=best_model_path,
                             algo=args.algo,
                             loader=val_loader,
                             device=config.DEVICE)
    else:
        print(f"Best model not found at {best_model_path}. Evaluation skipped.")
        # フォールバックとして、最終エポックのモデルを評価することも検討可能
        # last_epoch_path = save_root / f'epoch_{config.EPOCHS:03d}.pth'
        # if last_epoch_path.exists():
        #     print(f"Attempting to evaluate last epoch model: {last_epoch_path}")
        #     evaluate_final_model(model_path=last_epoch_path, ...)

if __name__ == '__main__':
    main()
