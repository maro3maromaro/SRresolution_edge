# ================================================================
# model.py  (自己完結ファクトリ / *args, **kwargs 対応) 2025‑04修正 ★デバッグプリント追加★
# ================================================================
"""超解像モデルの定義と構築を行うモジュール。

BasicSRライブラリが存在する場合はそれを利用し、存在しない場合は
簡易的なフォールバック実装を使用します。
`build_model`関数を通じて、指定された名前のモデルを構築します。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# try‑import ヘルパ
# ------------------------------------------------------------

def _import_or_fallback(module_path: str, cls_name: str, fallback_cls: type) -> type:
    """指定されたモジュールからクラスをインポートし、失敗した場合はフォールバッククラスを返します。

    Args:
        module_path (str): インポート試行対象のモジュールパス (例: 'basicsr.archs.edsr_arch')。
        cls_name (str): インポート試行対象のクラス名 (例: 'EDSR')。
        fallback_cls (type): インポート失敗時に使用するフォールバッククラス。

    Returns:
        type: インポートされたクラス、またはフォールバッククラス。
    """
    try:
        module = __import__(module_path, fromlist=[cls_name])
        actual_class = getattr(module, cls_name)
        print(f"[DEBUG] Successfully imported '{cls_name}' from '{module_path}'. Type: {type(actual_class)}")
        return actual_class
    except Exception as e:
        print(f"[DEBUG] Failed to import '{cls_name}' from '{module_path}', using fallback '{fallback_cls.__name__}'. Error: {e}")
        return fallback_cls

# ------------------------------------------------------------
# 共通小ブロック
# ------------------------------------------------------------
class _ResBlock(nn.Module):
    """基本的な残差ブロック。

    畳み込み層 -> ReLU -> 畳み込み層 の構造を持ち、入力との残差接続を行います。

    Attributes:
        conv1 (nn.Conv2d): 1番目の畳み込み層。
        relu (nn.ReLU): ReLU活性化関数。
        conv2 (nn.Conv2d): 2番目の畳み込み層。
    """
    def __init__(self, nf: int):
        """
        Args:
            nf (int): 畳み込み層のフィルタ数（特徴マップのチャネル数）。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        return x + self.conv2(self.relu(self.conv1(x)))

# ------------------------------------------------------------
# Fallback 実装
# ------------------------------------------------------------
class _FallbackEDSR(nn.Module):
    """EDSRモデルの簡易的なフォールバック実装。

    BasicSRのEDSRアーキテクチャを模倣した基本的な構造です。
    入力チャネル数、出力チャネル数、特徴マップ数、残差ブロック数を指定できます。
    upscale=1 (画質改善タスク) を想定しています。
    """
    def __init__(self, *args, num_in_ch: int = 1, num_out_ch: int = 1, nf: int = 64, nb: int = 8, upscale: int = 1, **kwargs):
        """
        Args:
            *args: BasicSRのEDSRが受け取る可能性のある他の位置引数 (無視されます)。
            num_in_ch (int, optional): 入力画像のチャネル数。デフォルトは1。
            num_out_ch (int, optional): 出力画像のチャネル数。デフォルトは1。
            nf (int, optional): 内部の特徴マップのチャネル数。デフォルトは64。
            nb (int, optional): 残差ブロックの数。デフォルトは8。
            upscale (int, optional): アップスケールファクタ。このフォールバックでは主に1を想定。デフォルトは1。
            **kwargs: BasicSRのEDSRが受け取る可能性のある他のキーワード引数 (無視されます)。
        """
        super().__init__()
        print(f"[DEBUG] _FallbackEDSR initialized with num_in_ch={num_in_ch}, num_out_ch={num_out_ch}, upscale={upscale}, nf={nf}, nb={nb}")
        self.head = nn.Conv2d(num_in_ch, nf, 3,1,1)
        self.body = nn.Sequential(*[_ResBlock(nf) for _ in range(nb)])
        self.tail = nn.Conv2d(nf, num_out_ch, 3,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル (B, num_in_ch, H, W)。

        Returns:
            torch.Tensor: 出力テンソル (B, num_out_ch, H, W)。
        """
        f = self.head(x)
        f_body = self.body(f)
        f_res = f_body + f # EDSRの主要な残差接続
        return self.tail(f_res)

class _EdgeFormerBlock(nn.Module):
    """EdgeFormerで提案されたようなアテンションベースのブロックの簡易フォールバック実装。

    畳み込み層とMultiheadAttentionを組み合わせた構造です。
    """
    def __init__(self, dim: int = 64, num_heads: int = 4):
        """
        Args:
            dim (int, optional): 入出力特徴マップのチャネル数 (アテンションのembed_dim)。デフォルトは64。
            num_heads (int, optional): MultiheadAttentionのヘッド数。デフォルトは4。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3,1,1)
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.conv2 = nn.Conv2d(dim, dim,3,1,1)
        print(f"[DEBUG] _EdgeFormerBlock initialized with dim={dim}, num_heads={num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル (B, C, H, W)。

        Returns:
            torch.Tensor: 出力テンソル (B, C, H, W)。
        """
        b,c,h,w = x.shape
        y = self.conv1(x)
        y_permuted = y.view(b,c,h*w).permute(0,2,1) # (B, H*W, C) for MultiheadAttention
        y_attn, _ = self.attn(y_permuted, y_permuted, y_permuted)
        y_restored = y_attn.permute(0,2,1).view(b,c,h,w) # (B, C, H, W)
        return x + self.conv2(y_restored) # Residual connection

class _FallbackRDN(nn.Module):
    """RDNモデルの非常に簡易的なフォールバック実装。

    BasicSRのRDNとは構造が大きく異なります。
    複数の残差ブロックをシーケンシャルに接続したものです。
    """
    def __init__(self, *args, num_in_ch: int = 1, num_out_ch: int = 1, nf: int = 64, num_dense_block: int = 4, growth_rate: int = 32, upscale: int = 1, **kwargs):
        """
        Args:
            *args: 他の位置引数 (無視されます)。
            num_in_ch (int, optional): 入力チャネル数。デフォルトは1。
            num_out_ch (int, optional): 出力チャネル数。デフォルトは1。
            nf (int, optional): 特徴マップ数。デフォルトは64。
            num_dense_block (int, optional): RDNのRDB (Residual Dense Block) に相当するブロック数。
                                          このフォールバックでは単純なResBlockの数。デフォルトは4。
            growth_rate (int, optional): RDNの成長率。このフォールバックでは未使用。デフォルトは32。
            upscale (int, optional): アップスケールファクタ。デフォルトは1。
            **kwargs: 他のキーワード引数 (無視されます)。
        """
        super().__init__()
        print(f"[DEBUG] _FallbackRDN initialized with num_in_ch={num_in_ch}, num_out_ch={num_out_ch}, upscale={upscale}, nf={nf}, num_dense_block={num_dense_block}")
        self.fea = nn.Conv2d(num_in_ch,nf,3,1,1)
        self.blocks=nn.ModuleList([_ResBlock(nf) for _ in range(num_dense_block)])
        self.outc = nn.Conv2d(nf,num_out_ch,3,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        f=self.fea(x)
        # 本来のRDNはより複雑な特徴融合（例：Global Feature Fusion, Local Feature Fusion）を行う
        for blk in self.blocks:
            f=blk(f) # このフォールバックでは単純なシーケンシャル処理
        return self.outc(f)

class _FallbackSRMD(nn.Module):
    """SRMDモデルの非常に簡易的なフォールバック実装。

    単純な畳み込み層とReLUのシーケンスです。
    """
    def __init__(self, *args, num_in_ch: int = 1, num_out_ch: int = 1, nf: int = 64, num_blocks: int = 5, upscale: int = 1, **kwargs):
        """
        Args:
            *args: 他の位置引数 (無視されます)。
            num_in_ch (int, optional): 入力チャネル数。デフォルトは1。
            num_out_ch (int, optional): 出力チャネル数。デフォルトは1。
            nf (int, optional): 特徴マップ数。デフォルトは64。
            num_blocks (int, optional): 中間畳み込みブロックの数。デフォルトは5。
            upscale (int, optional): アップスケールファクタ。デフォルトは1。
            **kwargs: 他のキーワード引数 (無視されます)。
        """
        super().__init__()
        print(f"[DEBUG] _FallbackSRMD initialized with num_in_ch={num_in_ch}, num_out_ch={num_out_ch}, upscale={upscale}, nf={nf}, num_blocks={num_blocks}")
        layers=[nn.Conv2d(num_in_ch,nf,3,1,1),nn.ReLU(True)]
        for _ in range(num_blocks):
            layers+=[nn.Conv2d(nf,nf,3,1,1),nn.ReLU(True)]
        layers+=[nn.Conv2d(nf,num_out_ch,3,1,1)]
        self.net=nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        return self.net(x)

class _FallbackNAFNet(nn.Module):
    """NAFNetモデルの非常に簡易的なフォールバック実装。

    U-Net風のエンコーダ・デコーダ構造です。BasicSRのNAFNetとは大きく異なります。
    """
    def __init__(self, *args, img_channel: int = 1, width: int = 32, middle_blk_num: int = 1, enc_blk_nums: list[int] | None = None, dec_blk_nums: list[int] | None = None, upscale: int = 1, **kwargs):
        """
        Args:
            *args: 他の位置引数 (無視されます)。
            img_channel (int, optional): 入出力画像のチャネル数。デフォルトは1。
            width (int, optional): ベースとなる特徴マップのチャネル数。デフォルトは32。
            middle_blk_num (int, optional): U-Netのボトルネック部分のブロック数。このフォールバックでは未使用。デフォルトは1。
            enc_blk_nums (list[int] | None, optional): エンコーダ各ステージのブロック数。このフォールバックでは固定。デフォルトはNone。
            dec_blk_nums (list[int] | None, optional): デコーダ各ステージのブロック数。このフォールバックでは固定。デフォルトはNone。
            upscale (int, optional): アップスケールファクタ。デフォルトは1。
            **kwargs: 他のキーワード引数 (無視されます)。
        """
        super().__init__()
        print(f"[DEBUG] _FallbackNAFNet initialized with img_channel={img_channel}, width={width}, upscale={upscale}")
        # enc_blk_nums, dec_blk_nums はこの簡易実装では固定的な層数として解釈
        
        # Simplified encoder
        self.enc1=nn.Conv2d(img_channel,width,3,1,1)       # -> width
        self.enc2=nn.Conv2d(width,width*2,3,2,1)           # Downsample -> width*2
        self.enc3=nn.Conv2d(width*2,width*4,3,2,1)         # Downsample -> width*4

        # Simplified decoder
        self.dec3=nn.ConvTranspose2d(width*4,width*2,2,2)  # Upsample -> width*2
        self.dec2=nn.ConvTranspose2d(width*2,width,2,2)    # Upsample -> width
        self.outc=nn.Conv2d(width,img_channel,3,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        # Encoder
        x1 = F.relu(self.enc1(x))    # (B, width, H, W)
        x2 = F.relu(self.enc2(x1))   # (B, width*2, H/2, W/2)
        x3 = F.relu(self.enc3(x2))   # (B, width*4, H/4, W/4)
        
        # Decoder with skip connections
        d3_out = F.relu(self.dec3(x3)) # (B, width*2, H/2, W/2)
        # スキップ接続: チャンネル数と空間サイズが一致する場合のみ加算
        if d3_out.shape == x2.shape:
            d3_out = d3_out + x2
        else: # サイズが異なる場合は単純なリサイズ (フォールバック用)
            if d3_out.shape[1] == x2.shape[1]: # チャンネル数が同じなら空間リサイズ
                 d3_out = F.interpolate(d3_out, size=x2.shape[2:], mode='bilinear', align_corners=False) + x2
            # チャンネル数も異なる場合は、この簡易フォールバックではスキップ接続を諦めるか、より複雑なアダプタが必要

        d2_out = F.relu(self.dec2(d3_out)) # (B, width, H, W)
        if d2_out.shape == x1.shape:
            d2_out = d2_out + x1
        else:
            if d2_out.shape[1] == x1.shape[1]:
                d2_out = F.interpolate(d2_out, size=x1.shape[2:], mode='bilinear', align_corners=False) + x1
                
        return self.outc(d2_out)

# ------------------------------------------------------------
# Try import BasicSR
# ------------------------------------------------------------
EDSR   = _import_or_fallback('basicsr.archs.edsr_arch','EDSR', _FallbackEDSR)
RDN    = _import_or_fallback('basicsr.archs.rdn_arch','RDN',  _FallbackRDN)
SRMD   = _import_or_fallback('basicsr.archs.srmd_arch','SRMD', _FallbackSRMD)
SwinIR = _import_or_fallback('basicsr.archs.swinir_arch','SwinIR', _FallbackEDSR) # SwinIRのFallbackはEDSRに注意
NAFNet = _import_or_fallback('basicsr.archs.nafnet_arch','NAFNet', _FallbackNAFNet)
EdgeFormerBlock = _import_or_fallback('basicsr.archs.edgeformer_arch','EdgeFormerBlock',_EdgeFormerBlock) # カスタムブロック想定

# ------------------------------------------------------------
# 派生モデル
# ------------------------------------------------------------

def _build_edgeformer_edsr() -> nn.Module:
    """EdgeFormer風ブロックをEDSRモデルの末尾数ブロックに組み込んだモデルを構築します。

    EDSRモデルをベースとし、そのbody部分の最後の4ブロックを`_EdgeFormerBlock`に置き換えます。
    BasicSRのEDSRまたはフォールバックEDSRが使用されます。

    Returns:
        nn.Module: 構築されたEdgeFormer-EDSRモデル。
    
    Raises:
        TypeError: EDSRモデルの初期化に失敗した場合。
    """
    print(f"[DEBUG] _build_edgeformer_edsr called. EDSR type: {type(EDSR)}")
    try:
        # BasicSRのEDSRが期待する標準的な引数で試行
        base = EDSR(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=16, upscale=1) # nbを16などに増やすことが多い
    except TypeError as e:
        print(f"[DEBUG] EDSR instantiation with num_feat, num_block failed: {e}. Trying with minimal args.")
        # より基本的な引数で再試行 (フォールバックEDSRや古いBasicSRバージョン用)
        base = EDSR(num_in_ch=1, num_out_ch=1, nf=64, nb=16, upscale=1) # nf, nbはフォールバック用

    if hasattr(base,'body') and isinstance(base.body, nn.Sequential) and len(base.body) >= 4:
        num_features = 64 # デフォルト
        # EDSRインスタンスから実際の特徴マップ数を取得する試み
        if hasattr(base, 'num_feat'): # BasicSRのEDSR (v1.3.5など)
            num_features = base.num_feat
        elif hasattr(base, 'conv_first') and isinstance(base.conv_first, nn.Conv2d): # EDSR (一部のバージョン)
             num_features = base.conv_first.out_channels
        elif hasattr(base, 'head') and isinstance(base.head, nn.Conv2d): # _FallbackEDSR
            num_features = base.head.out_channels
        
        print(f"[DEBUG] Replacing last 4 blocks of EDSR body with EdgeFormerBlock (dim={num_features})")
        for i in range(-4, 0): # bodyの最後の4ブロックを置き換える
            actual_index = i + len(base.body)
            if actual_index >= 0 :
                 base.body[actual_index] = EdgeFormerBlock(dim=num_features)
            else:
                print(f"[DEBUG] Invalid index {actual_index} for body replacement.")
    else:
        body_info = "None"
        if hasattr(base, 'body'):
            body_info = f"Type: {type(base.body)}, Length: {len(base.body) if isinstance(base.body, nn.Sequential) else 'N/A'}"
        print(f"[DEBUG] EDSR base model does not have 'body' attribute as expected, or body is not nn.Sequential / too short. Body info: {body_info}")
    return base

class _RDN_Edge(nn.Module):
    """RDNモデルをバックボーンとし、エッジ関連の処理（ヘッド）を追加したモデルのフォールバック実装。

    この実装では、エッジヘッドの出力は現在使用されていません。
    """
    def __init__(self):
        super().__init__()
        # BasicSRのRDNまたはフォールバックRDNを使用
        # 引数はBasicSRのRDNに合わせる (G0, RDNBs, c, G, upscale)
        # フォールバックの場合は nf, num_dense_block など
        try:
            self.backbone = RDN(num_in_ch=1, num_out_ch=1, num_feat=64, num_blocks=4, upscale=1) # num_blocksはRDBの数
        except TypeError: # フォールバックRDN用の引数
             self.backbone = RDN(num_in_ch=1, num_out_ch=1, nf=64, num_dense_block=4, upscale=1)
        
        self.num_feat_backbone = 64 # バックボーンの特徴量に合わせて調整
        if hasattr(self.backbone, 'num_feat'):
            self.num_feat_backbone = self.backbone.num_feat
        elif hasattr(self.backbone, 'fea') and isinstance(self.backbone.fea, nn.Conv2d): # _FallbackRDN
            self.num_feat_backbone = self.backbone.fea.out_channels

        self.edge_head = nn.Conv2d(self.num_feat_backbone, 1, 1) # バックボーンの出力チャネル数に合わせる

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: バックボーンからの出力テンソル。
        """
        # RDNのフォワードパス。BasicSRのRDNは複雑な特徴抽出と融合を行う。
        # フォールバックRDNは単純なシーケンシャル処理。
        features = self.backbone(x) # RDNの最終出力または中間特徴量を取得
        
        # edge_headの適用方法:
        # 1. RDNの最終出力 `features` に適用する場合 (チャンネル数が合えば)
        #    if features.size(1) == self.edge_head.in_channels:
        #        edge_output = self.edge_head(features)
        #    else: # チャンネル数が合わない場合は、何らかの処理が必要
        #        # print(f"Warning: Channel mismatch for edge_head. Backbone out: {features.size(1)}, Edge head in: {self.edge_head.in_channels}")
        #        pass # またはエラー
        # 2. RDNの内部特徴量に適用する場合 (より複雑な改造が必要)

        # この実装では、edge_headの出力は明示的には返していない。
        # _ = self.edge_head(features_for_edge_head) # のような形を想定していたが、
        # features_for_edge_head をどこから取るかが問題。
        # ここでは、バックボーンの最終出力をそのまま返す。
        return features


# ------------------------------------------------------------
# build_model ファクトリ
# ------------------------------------------------------------

def build_model(name:str) -> nn.Module:
    """指定された名前の超解像モデルを構築して返します。

    Args:
        name (str): 構築するモデルの名前。
            'edgeformer_edsr', 'rdn_edge', 'srmd', 'swinir_light', 'nafnet' など。

    Returns:
        nn.Module: 構築されたPyTorchモデル。

    Raises:
        ValueError: 指定されたモデル名が存在しない場合。
    """
    name = name.lower()
    print(f"[DEBUG] build_model called for: {name}")
    if name=='edgeformer_edsr':
        return _build_edgeformer_edsr()
    if name=='rdn_edge':
        return _RDN_Edge()
    if name=='srmd':
        # BasicSRのSRMDの標準的な引数 (num_in_ch, num_out_ch, num_feat, num_block, upscale)
        return SRMD(num_in_ch=1,num_out_ch=1, num_feat=64, num_block=12, upscale=1) # num_blockはSRMDの仕様に合わせる
    if name=='swinir_light':
        # SwinIRの引数例 (Light版を想定)
        # SwinIRのフォールバックが_FallbackEDSRの場合、これらの引数の多くは無視される
        return SwinIR(upscale=1, img_size=256, # img_sizeはパッチサイズに合わせる
                      window_size=8, img_range=1., depths=[6, 6, 6, 6],
                      embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='') # upsampler='' for SRx1
    if name=='nafnet':
        # NAFNetの引数例 (img_channel, widthなど)
        return NAFNet(img_channel=1, width=32, middle_blk_num=1, 
                      enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1], upscale=1) # upscale=1
    raise ValueError(f'Unknown model name: {name}')
