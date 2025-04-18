# ================================================================
# model.py  (自己完結ファクトリ / *args, **kwargs 対応) 2025‑04修正
# ================================================================
#  BasicSR が無い環境でも動くよう、try‑import → fallback 方式。
#  Fallback 実装は **不明キーワードを無視** できるように
#  `__init__(self, *args, **kwargs)` で受け取ります。
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# try‑import ヘルパ
# ------------------------------------------------------------

def _import_or_fallback(module_path: str, cls_name: str, fallback_cls):
    try:
        module = __import__(module_path, fromlist=[cls_name])
        return getattr(module, cls_name)
    except Exception:
        return fallback_cls

# ------------------------------------------------------------
# 共通小ブロック
# ------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
    def forward(self,x):
        return x + self.conv2(self.relu(self.conv1(x)))

# ------------------------------------------------------------
# Fallback 実装
# ------------------------------------------------------------
class _FallbackEDSR(nn.Module):
    def __init__(self,*args,num_in_ch=1,num_out_ch=1,nf=64,nb=8,**kwargs):
        super().__init__()
        self.head = nn.Conv2d(num_in_ch, nf, 3,1,1)
        self.body = nn.Sequential(*[_ResBlock(nf) for _ in range(nb)])
        self.tail = nn.Conv2d(nf, num_out_ch, 3,1,1)
    def forward(self,x):
        f = self.head(x)
        f = self.body(f)+f
        return self.tail(f)

class _EdgeFormerBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3,1,1)
        self.attn  = nn.MultiheadAttention(dim,4,batch_first=True)
        self.conv2 = nn.Conv2d(dim, dim,3,1,1)
    def forward(self,x):
        b,c,h,w=x.shape
        y=self.conv1(x).view(b,c,h*w).permute(0,2,1)
        y,_=self.attn(y,y,y)
        y=y.permute(0,2,1).view(b,c,h,w)
        return x+self.conv2(y)

class _FallbackRDN(nn.Module):
    def __init__(self,*args,num_in_ch=1,num_out_ch=1,nf=64,num_dense=4,**kwargs):
        super().__init__()
        self.fea = nn.Conv2d(num_in_ch,nf,3,1,1)
        self.blocks=nn.ModuleList([_ResBlock(nf) for _ in range(num_dense)])
        self.outc = nn.Conv2d(nf,num_out_ch,3,1,1)
    def forward(self,x):
        f=self.fea(x)
        for blk in self.blocks:
            f=blk(f)
        return self.outc(f)

class _FallbackSRMD(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        layers=[nn.Conv2d(1,64,3,1,1),nn.ReLU(True)]
        for _ in range(5):
            layers+=[nn.Conv2d(64,64,3,1,1),nn.ReLU(True)]
        layers+=[nn.Conv2d(64,1,3,1,1)]
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

class _FallbackNAFNet(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.enc1=nn.Conv2d(1,32,3,1,1)
        self.enc2=nn.Conv2d(32,64,3,2,1)
        self.enc3=nn.Conv2d(64,128,3,2,1)
        self.dec3=nn.ConvTranspose2d(128,64,2,2)
        self.dec2=nn.ConvTranspose2d(64,32,2,2)
        self.outc=nn.Conv2d(32,1,3,1,1)
    def forward(self,x):
        x1=F.relu(self.enc1(x)); x2=F.relu(self.enc2(x1)); x3=F.relu(self.enc3(x2))
        d3=F.relu(self.dec3(x3))+x2
        d2=F.relu(self.dec2(d3))+x1
        return self.outc(d2)

# ------------------------------------------------------------
# Try import BasicSR
# ------------------------------------------------------------
EDSR   = _import_or_fallback('basicsr.archs.edsr_arch','EDSR', _FallbackEDSR)
RDN    = _import_or_fallback('basicsr.archs.rdn_arch','RDN',  _FallbackRDN)
SRMD   = _import_or_fallback('basicsr.archs.srmd_arch','SRMD', _FallbackSRMD)
SwinIR = _import_or_fallback('basicsr.archs.swinir_arch','SwinIR', _FallbackEDSR)
NAFNet = _import_or_fallback('basicsr.archs.nafnet_arch','NAFNet', _FallbackNAFNet)
EdgeFormerBlock=_import_or_fallback('basicsr.archs.edgeformer_arch','EdgeFormerBlock',_EdgeFormerBlock)

# ------------------------------------------------------------
# 派生モデル
# ------------------------------------------------------------

def _build_edgeformer_edsr():
    base = EDSR(num_in_ch=1,num_out_ch=1,upscale=1)
    if hasattr(base,'body') and len(base.body)>=4:
        for i in range(-4,0):
            base.body[i]=EdgeFormerBlock(dim=64)
    return base

class _RDN_Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RDN(num_in_ch=1,num_out_ch=1,upscale=1)
        self.edge_head = nn.Conv2d(64,1,1)
    def forward(self,x):
        if hasattr(self.backbone,'fea'):
            f=self.backbone.fea(x)
            for blk in self.backbone.blocks:
                f=blk(f)
            out=self.backbone.outc(f)
        else:
            out=self.backbone(x)
        _=self.edge_head(out)
        return out

# ------------------------------------------------------------
# build_model ファクトリ
# ------------------------------------------------------------

def build_model(name:str):
    name=name.lower()
    if name=='edgeformer_edsr':
        return _build_edgeformer_edsr()
    if name=='rdn_edge':
        return _RDN_Edge()
    if name=='srmd':
        return SRMD(num_in_ch=1,num_out_ch=1)
    if name=='swinir_light':
        return SwinIR(num_in_ch=1,num_out_ch=1,upscale=1)
    if name=='nafnet':
        return NAFNet()
    raise ValueError(f'Unknown model {name}')
