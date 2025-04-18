# ================================================================
# model.py  (モデル組み立て)
# ================================================================
#  BasicSR で提供される各種アーキテクチャをラップし、
#  "algo" 文字列 → PyTorch nn.Module を返す関数 build_model() を実装。
# ------------------------------------------------
# 追加機能：
#   * EdgeFormer‑EDSR  
#       - EDSR 本体の最後の 4 ブロックを EdgeFormerBlock に置換
#   * RDN + EdgeHead   
#       - RDN の特徴マップから 1x1 Conv で EdgeMap を同時回帰
#   * 既存モデル (SRMD / SwinIR‑light / NAFNet) をそのまま呼び出し
# ================================================================

import torch.nn as nn

# --- BasicSR アーキテクチャを import ---
from basicsr.archs.rdn_arch import RDN
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.edgeformer_arch import EdgeFormerBlock  # EdgeFormer 実装を配置しておく
from basicsr.archs.srmd_arch import SRMD
from basicsr.archs.swinir_arch import SwinIR
from basicsr.archs.nafnet_arch import NAFNet

# ------------------------------------------------------------
# EdgeFormer‑EDSR
# ------------------------------------------------------------

def _wrap_edgeformer_edsr():
    """EDSR の末尾 Residual Block を EdgeFormerBlock に置換した派生モデル"""
    base = EDSR(num_in_ch=1, num_out_ch=1, upscale=1)
    # body[-4:] を EdgeFormerBlock に置換 (dim=64 は EDSR デフォルト)
    for i in range(-4, 0):
        base.body[i] = EdgeFormerBlock(dim=64)
    return base

# ------------------------------------------------------------
# RDN + Edge Head
# ------------------------------------------------------------

def _wrap_rdn_edge():
    """RDN にエッジマップ出力を付加 (学習時のみ使用)"""
    class RDN_Edge(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdn = RDN(num_in_ch=1, num_out_ch=1, upscale=1)
            # 64ch → 1ch で EdgeMap を生成 (損失に利用可)
            self.edge_head = nn.Conv2d(64, 1, 1)
        def forward(self, x):
            feat = self.rdn.fea_conv(x)  # 初段 Conv
            res  = self.rdn.body(feat)   # RDN 本体
            out  = self.rdn.upsampler(res + feat)  # SR 出力
            edge = self.edge_head(res)             # EdgeMap (返さず学習時にフック)
            return out
    return RDN_Edge()

# ------------------------------------------------------------
# エイリアス関数 build_model(algo)
# ------------------------------------------------------------

def build_model(name: str):
    """algo 文字列 → インスタンスを返すファクトリ関数"""
    name = name.lower()
    if name == 'edgeformer_edsr':
        return _wrap_edgeformer_edsr()
    if name == 'rdn_edge':
        return _wrap_rdn_edge()
    if name == 'srmd':
        return SRMD(num_in_ch=1, num_out_ch=1, upscale=1)
    if name == 'swinir_light':
        # window_size=8, embed_dim=60 など軽量設定
        return SwinIR(upscale=1, img_size=256, window_size=8,
                      depths=[6]*6, embed_dim=60, num_heads=[6]*6,
                      mlp_ratio=2, num_in_ch=1, num_out_ch=1)
    if name == 'nafnet':
        return NAFNet(num_in_ch=1, num_out_ch=1)
    raise ValueError(f'Unknown model {name}')
