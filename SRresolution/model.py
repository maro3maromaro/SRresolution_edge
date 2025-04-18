# ================================================================
# model.py  (自己完結型モデル組み立てモジュール)
# ================================================================
#  BasicSR をインストールしていない／サブセットしか無い環境でも動くよう、
#  インポートに失敗したアーキテクチャは **簡易実装 (fallback)** をこの
#  ファイル内で自動生成します。
# ---------------------------------------------------------------
#  提供アルゴリズム:
#    * edgeformer_edsr : EDSR の tail を EdgeFormerBlock に置換
#    * rdn_edge        : 簡易 RDN + Edge Head
#    * srmd            : 簡易 SRMD (Blur/Noise 入力なし版)
#    * swinir_light    : SwinIR が import できなければ簡易 CNN にフォールバック
#    * nafnet          : 簡易 NAFNet ライクネット
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1) Utility: 試しに import → 失敗したら自前実装
# ------------------------------------------------------------

def _import_or_define(module_path: str, cls_name: str, fallback_fn):
    """module_path.cls_name を import し、失敗したら fallback_fn() を返す"""
    try:
        module = __import__(module_path, fromlist=[cls_name])
        return getattr(module, cls_name)
    except (ImportError, AttributeError):
        return fallback_fn()

# ------------------------------------------------------------
# 2) Fallback 実装群
# ------------------------------------------------------------

# -- EdgeFormerBlock (簡易版: Conv → Self‑Attention → Conv) --
class _EdgeFormerBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.attn  = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
    def forward(self, x):
        res = x
        x = self.conv1(x)
        b,c,h,w = x.shape
        x_flat = x.view(b, c, h*w).permute(0,2,1)
        x_attn,_ = self.attn(x_flat, x_flat, x_flat)
        x_attn = x_attn.permute(0,2,1).view(b,c,h,w)
        x = self.conv2(x_attn) + res
        return x

# -- EDSR (fallback: 16 Residual Blocks + Upscale=1) --
class _FallbackEDSR(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, nf=64, nb=16):
        super().__init__()
        self.head = nn.Conv2d(num_in_ch, nf, 3, 1, 1)
        body = []
        for _ in range(nb):
            body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(inplace=True),
                     nn.Conv2d(nf, nf, 3, 1, 1)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(nf, num_out_ch, 3, 1, 1)
    def forward(self, x):
        feat = self.head(x)
        res  = self.body(feat) + feat
        out  = self.tail(res)
        return out

# -- RDN (fallback: 4 Dense Blocks) --
class _FallbackRDN(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, nf=64, num_blocks=4):
        super().__init__()
        self.fea_conv = nn.Conv2d(num_in_ch, nf, 3, 1, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nf, nf, 3,1,1), nn.ReLU(True),
                nn.Conv2d(nf, nf, 3,1,1), nn.ReLU(True))
            for _ in range(num_blocks)])
        self.upsampler = nn.Conv2d(nf, num_out_ch, 3,1,1)
    def forward(self, x):
        x = self.fea_conv(x)
        res = x
        for blk in self.blocks:
            res = res + blk(res)
        return self.upsampler(res)

# -- SRMD (fallback: Shallow CNN) --
class _FallbackSRMD(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1):
        super().__init__()
        layers = [nn.Conv2d(num_in_ch, 64, 3,1,1), nn.ReLU(True)]
        for _ in range(7):
            layers += [nn.Conv2d(64,64,3,1,1), nn.ReLU(True)]
        layers += [nn.Conv2d(64, num_out_ch, 3,1,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# -- SwinIR-light fallback: use EDSR small --
_FallbackSwin = _FallbackEDSR

# -- NAFNet (fallback: UNet‑like) --
class _FallbackNAFNet(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1):
        super().__init__()
        self.enc1 = nn.Conv2d(num_in_ch, 32, 3,1,1)
        self.enc2 = nn.Conv2d(32, 64, 3,2,1)
        self.enc3 = nn.Conv2d(64,128, 3,2,1)
        self.dec3 = nn.ConvTranspose2d(128,64,2,2)
        self.dec2 = nn.ConvTranspose2d(64,32,2,2)
        self.outc = nn.Conv2d(32, num_out_ch,3,1,1)
    def forward(self,x):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        d3 = F.relu(self.dec3(x3)) + x2
        d2 = F.relu(self.dec2(d3)) + x1
        out= self.outc(d2)
        return out

# ------------------------------------------------------------
# 3) Try importing actual BasicSR implementations, else fallback
# ------------------------------------------------------------

EDSR = _import_or_define('basicsr.archs.edsr_arch', 'EDSR', lambda: _FallbackEDSR)
RDN  = _import_or_define('basicsr.archs.rdn_arch',  'RDN',  lambda: _FallbackRDN)
SRMD = _import_or_define('basicsr.archs.srmd_arch', 'SRMD', lambda: _FallbackSRMD)
SwinIR = _import_or_define('basicsr.archs.swinir_arch', 'SwinIR', lambda: _FallbackSwin)
NAFNet = _import_or_define('basicsr.archs.nafnet_arch', 'NAFNet', lambda: _FallbackNAFNet)
EdgeFormerBlock = _import_or_define('basicsr.archs.edgeformer_arch', 'EdgeFormerBlock', lambda: _EdgeFormerBlock)

# ------------------------------------------------------------
# 4) 派生モデル生成関数
# ------------------------------------------------------------

def _wrap_edgeformer_edsr():
    """EDSR (or fallback) の末尾 4 ブロックを EdgeFormerBlock へ置換"""
    base = EDSR(num_in_ch=1, num_out_ch=1, upscale=1)
    if hasattr(base, 'body') and len(base.body) >= 4:
        for i in range(-4, 0):
            base.body[i] = EdgeFormerBlock(dim=64)
    return base

class _RDN_Edge(nn.Module):
    """RDN + Edge Head (Edge Map は訓練用; 推論時は破棄)"""
    def __init__(self):
        super().__init__()
        self.rdn = RDN(num_in_ch=1, num_out_ch=1, upscale=1)
        self.edge_head = nn.Conv2d(64,1,1)
    def forward(self, x):
        # RDN fallback は fea_conv/body/upsampler 構造を想定
        feat = self.rdn.fea_conv(x) if hasattr(self.rdn,'fea_conv') else self.rdn(x)
        if hasattr(self.rdn,'body'):
            res  = self.rdn.body(feat)
            out  = self.rdn.upsampler(res + feat)
        else:
            out = feat
        _ = self.edge_head(feat)  # EdgeMap (無視)
        return out

# ------------------------------------------------------------
# 5) build_model: 文字列 → インスタンス
# ------------------------------------------------------------

def build_model(name: str):
    name = name.lower()
    if name == 'edgeformer_edsr':
        return _wrap_edgeformer_edsr()
    if name == 'rdn_edge':
        return _RDN_Edge()
    if name == 'srmd':
        return SRMD(num_in_ch=1, num_out_ch=1)
    if name == 'swinir_light':
        return SwinIR(num_in_ch=1, num_out_ch=1, upscale=1) if SwinIR is not _FallbackSwin else SwinIR()
    if name == 'nafnet':
        return NAFNet(num_in_ch=1, num_out_ch=1)
    raise ValueError(f'Unknown model name: {name}')
