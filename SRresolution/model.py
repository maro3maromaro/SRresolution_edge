# ================================================================
# model.py (自己完結ファクトリ / 軽量モデル対応) ★docstring付き★
# ================================================================
"""超解像モデルの定義と構築を行うモジュール。

BasicSRライブラリが存在する場合はそれを利用し、存在しない場合は
簡易的なフォールバック実装を使用します。
`build_model`関数を通じて、指定された名前のモデルを構築します。
config.pyのMODEL_CONFIGSからパラメータを読み込み、軽量化に対応します。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config # プロジェクト共通設定をインポート

# ------------------------------------------------------------
# try‑import ヘルパ
# ------------------------------------------------------------
def _import_or_fallback(module_path: str, cls_name: str, fallback_cls: type) -> type:
    """指定されたモジュールからクラスをインポートし、失敗した場合はフォールバッククラスを返します。

    Args:
        module_path (str): インポート試行対象のモジュールパス。
        cls_name (str): インポート試行対象のクラス名。
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
    """基本的な残差ブロック。畳み込み層 -> ReLU -> 畳み込み層 + 入力。"""
    def __init__(self, nf: int):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))

# ------------------------------------------------------------
# 新しいシンプルな軽量モデル
# ------------------------------------------------------------
class SimpleSRNet(nn.Module):
    """非常にシンプルなCNNベースの超解像モデル（画質改善タスク用）。

    数層の畳み込み層とReLUで構成されます。
    """
    def __init__(self, num_in_ch: int = 1, num_out_ch: int = 1, nf: int = 32, nb: int = 3, upscale: int = 1, **kwargs):
        """
        Args:
            num_in_ch (int, optional): 入力チャネル数。デフォルトは1。
            num_out_ch (int, optional): 出力チャネル数。デフォルトは1。
            nf (int, optional): 中間層のチャネル数。デフォルトは32。
            nb (int, optional): 中間畳み込み層の数。デフォルトは3。
            upscale (int, optional): アップスケールファクタ (このモデルでは1を想定)。デフォルトは1。
            **kwargs: その他の引数 (無視されます)。
        """
        super().__init__()
        print(f"[DEBUG] SimpleSRNet initialized with num_in_ch={num_in_ch}, num_out_ch={num_out_ch}, nf={nf}, nb={nb}, upscale={upscale}")
        
        layers = [nn.Conv2d(num_in_ch, nf, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nb -1): # nbは総畳み込み層数 (head + body) と解釈。ここではbody部分。
            layers.append(nn.Conv2d(nf, nf, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(nf, num_out_ch, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 出力テンソル。
        """
        return self.net(x)

# ------------------------------------------------------------
# Fallback 実装 (デフォルト値を軽量化)
# ------------------------------------------------------------
class _FallbackEDSR(nn.Module):
    """EDSRモデルの簡易的なフォールバック実装 (軽量化デフォルト)。"""
    def __init__(self, *args, num_in_ch: int = 1, num_out_ch: int = 1, nf: int = 32, nb: int = 4, upscale: int = 1, **kwargs): # nf, nbを小さく
        super().__init__()
        # kwargsからnum_feat, num_blockを読み取ろうと試みる (BasicSRの引数名)
        # config.MODEL_CONFIGSからの値が優先されるようにbuild_model側で調整
        self.nf = kwargs.get('num_feat', nf)
        self.nb = kwargs.get('num_block', nb)
        
        print(f"[DEBUG] _FallbackEDSR initialized with num_in_ch={num_in_ch}, num_out_ch={num_out_ch}, nf(num_feat)={self.nf}, nb(num_block)={self.nb}, upscale={upscale}")
        self.head = nn.Conv2d(num_in_ch, self.nf, 3,1,1)
        self.body = nn.Sequential(*[_ResBlock(self.nf) for _ in range(self.nb)])
        self.tail = nn.Conv2d(self.nf, num_out_ch, 3,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.head(x)
        f_body = self.body(f)
        f_res = f_body + f
        return self.tail(f_res)

class _EdgeFormerBlock(nn.Module):
    """EdgeFormer風ブロックの簡易フォールバック実装。"""
    def __init__(self, dim: int = 32, num_heads: int = 4, **kwargs): # dimのデフォルトを小さく
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(self.dim, self.dim, 3,1,1)
        self.attn  = nn.MultiheadAttention(embed_dim=self.dim, num_heads=num_heads, batch_first=True)
        self.conv2 = nn.Conv2d(self.dim, self.dim,3,1,1)
        print(f"[DEBUG] _EdgeFormerBlock initialized with dim={self.dim}, num_heads={num_heads}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,c,h,w = x.shape
        y = self.conv1(x)
        y_permuted = y.view(b,c,h*w).permute(0,2,1)
        y_attn, _ = self.attn(y_permuted, y_permuted, y_permuted)
        y_restored = y_attn.permute(0,2,1).view(b,c,h,w)
        return x + self.conv2(y_restored)

class _FallbackRDN(nn.Module):
    """RDNモデルの非常に簡易的なフォールバック実装 (軽量化デフォルト)。"""
    def __init__(self, *args, num_in_ch: int = 1, num_out_ch: int = 1, nf: int = 32, num_dense_block: int = 3, growth_rate: int = 16, upscale: int = 1, **kwargs): # nf, num_dense_block, growth_rateを小さく
        super().__init__()
        self.nf = kwargs.get('num_feat', nf) # BasicSRのRDNの引数名 num_feat
        self.num_dense_block = kwargs.get('num_block', num_dense_block) # BasicSRのRDNのRDB数 num_block
        self.growth_rate = kwargs.get('growth_rate', growth_rate) # このフォールバックでは直接使わないが保持

        print(f"[DEBUG] _FallbackRDN initialized with num_in_ch={num_in_ch}, num_out_ch={num_out_ch}, nf(num_feat)={self.nf}, num_dense_block(num_block)={self.num_dense_block}, upscale={upscale}")
        self.fea = nn.Conv2d(num_in_ch, self.nf,3,1,1)
        # RDNはResidual Dense Block (RDB) を使うが、ここでは簡易的にResBlockを使用
        self.blocks=nn.ModuleList([_ResBlock(self.nf) for _ in range(self.num_dense_block)])
        self.outc = nn.Conv2d(self.nf,num_out_ch,3,1,1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f=self.fea(x)
        for blk in self.blocks:
            f=blk(f)
        return self.outc(f)

class _FallbackSRMD(nn.Module):
    """SRMDモデルの非常に簡易的なフォールバック実装 (軽量化デフォルト)。"""
    def __init__(self, *args, num_in_ch: int = 1, num_out_ch: int = 1, nf: int = 32, num_blocks: int = 4, upscale: int = 1, **kwargs): # nf, num_blocksを小さく
        super().__init__()
        self.nf = kwargs.get('num_feat', nf)
        self.num_blocks = kwargs.get('num_block', num_blocks)
        print(f"[DEBUG] _FallbackSRMD initialized with num_in_ch={num_in_ch}, num_out_ch={num_out_ch}, nf(num_feat)={self.nf}, num_blocks(num_block)={self.num_blocks}, upscale={upscale}")
        layers=[nn.Conv2d(num_in_ch,self.nf,3,1,1),nn.ReLU(True)]
        for _ in range(self.num_blocks):
            layers+=[nn.Conv2d(self.nf,self.nf,3,1,1),nn.ReLU(True)]
        layers+=[nn.Conv2d(self.nf,num_out_ch,3,1,1)]
        self.net=nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _FallbackNAFNet(nn.Module):
    """NAFNetモデルの非常に簡易的なフォールバック実装 (軽量化デフォルト)。"""
    def __init__(self, *args, img_channel: int = 1, width: int = 16, middle_blk_num: int = 1, enc_blk_nums: list[int] | None = None, dec_blk_nums: list[int] | None = None, upscale: int = 1, **kwargs): # widthを小さく
        super().__init__()
        self.width = width
        print(f"[DEBUG] _FallbackNAFNet initialized with img_channel={img_channel}, width={self.width}, upscale={upscale}")
        if enc_blk_nums is None: enc_blk_nums = [1, 1, 1]
        if dec_blk_nums is None: dec_blk_nums = [1, 1, 1]
        
        self.enc1=nn.Conv2d(img_channel,self.width,3,1,1)
        self.enc2=nn.Conv2d(self.width,self.width*2,3,2,1)
        self.enc3=nn.Conv2d(self.width*2,self.width*4,3,2,1)
        self.dec3=nn.ConvTranspose2d(self.width*4,self.width*2,2,2)
        self.dec2=nn.ConvTranspose2d(self.width*2,self.width,2,2)
        self.outc=nn.Conv2d(self.width,img_channel,3,1,1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1=F.relu(self.enc1(x)); x2=F.relu(self.enc2(x1)); x3=F.relu(self.enc3(x2))
        d3_out = F.relu(self.dec3(x3))
        if d3_out.shape == x2.shape: d3_out = d3_out + x2
        d2_out = F.relu(self.dec2(d3_out))
        if d2_out.shape == x1.shape: d2_out = d2_out + x1
        return self.outc(d2_out)

# ------------------------------------------------------------
# Try import BasicSR
# ------------------------------------------------------------
EDSR   = _import_or_fallback('basicsr.archs.edsr_arch','EDSR', _FallbackEDSR)
RDN    = _import_or_fallback('basicsr.archs.rdn_arch','RDN',  _FallbackRDN)
SRMD   = _import_or_fallback('basicsr.archs.srmd_arch','SRMD', _FallbackSRMD)
SwinIR = _import_or_fallback('basicsr.archs.swinir_arch','SwinIR', _FallbackEDSR) # SwinIRのフォールバックは簡易EDSR
NAFNet = _import_or_fallback('basicsr.archs.nafnet_arch','NAFNet', _FallbackNAFNet)
EdgeFormerBlock = _import_or_fallback('basicsr.archs.edgeformer_arch','EdgeFormerBlock',_EdgeFormerBlock)

# ------------------------------------------------------------
# 派生モデル構築関数
# ------------------------------------------------------------
def _build_edgeformer_edsr(params: dict) -> nn.Module:
    """EdgeFormer風ブロックをEDSRモデルに組み込んだモデルを構築します。"""
    print(f"[DEBUG] _build_edgeformer_edsr called with params: {params}. EDSR type: {type(EDSR)}")
    
    # EDSRの初期化。BasicSR版とフォールバック版で引数名が異なる可能性に対応
    # paramsから 'num_feat', 'num_block', 'upscale' などを取得
    # フォールバック用に 'nf', 'nb' も考慮
    edsr_params = {
        'num_in_ch': params.get('num_in_ch', 1),
        'num_out_ch': params.get('num_out_ch', 1),
        'num_feat': params.get('num_feat', 32), # 軽量化デフォルト
        'num_block': params.get('num_block', 8), # 軽量化デフォルト
        'upscale': params.get('upscale', 1)
    }
    # フォールバックEDSRが使われる場合、nfとnbにnum_featとnum_blockをマッピング
    if EDSR == _FallbackEDSR:
        edsr_params['nf'] = edsr_params.pop('num_feat')
        edsr_params['nb'] = edsr_params.pop('num_block')

    try:
        base = EDSR(**edsr_params)
    except TypeError as e:
        print(f"[DEBUG] EDSR instantiation failed with {edsr_params}: {e}. Trying minimal args for fallback.")
        # フォールバック用の最小引数で再試行
        base = EDSR(num_in_ch=1, num_out_ch=1, nf=32, nb=8, upscale=1)


    if hasattr(base,'body') and isinstance(base.body, nn.Sequential) and len(base.body) >= 4:
        edgeformer_dim = params.get('edgeformer_dim', edsr_params.get('num_feat', edsr_params.get('nf', 32)))
        print(f"[DEBUG] Replacing last 4 blocks of EDSR body with EdgeFormerBlock (dim={edgeformer_dim})")
        for i in range(-4, 0):
            actual_index = i + len(base.body)
            if actual_index >= 0 :
                 base.body[actual_index] = EdgeFormerBlock(dim=edgeformer_dim)
    else:
        body_info = "None"
        if hasattr(base, 'body'): body_info = f"Type: {type(base.body)}, Length: {len(base.body) if isinstance(base.body, nn.Sequential) else 'N/A'}"
        print(f"[DEBUG] EDSR base model does not have 'body' as expected. Body info: {body_info}")
    return base

class _RDN_Edge(nn.Module):
    """RDNモデルをバックボーンとし、エッジ関連の処理ヘッドを追加したモデル。"""
    def __init__(self, params: dict):
        super().__init__()
        print(f"[DEBUG] _RDN_Edge initialized with params: {params}")
        rdn_params = {
            'num_in_ch': params.get('num_in_ch', 1),
            'num_out_ch': params.get('num_out_ch', 1),
            'num_feat': params.get('num_feat', 32), # 軽量化デフォルト
            'num_block': params.get('num_block', 3), # RDBの数, 軽量化デフォルト
            'upscale': params.get('upscale', 1),
            # BasicSRのRDN特有の引数 (フォールバックでは一部無視される可能性あり)
            'num_dense_layer': params.get('num_dense_block', 4), # RDB内の畳み込み層数 (BasicSRでは Gc)
            'growth_rate': params.get('growth_rate', 16)    # BasicSRでは G
        }
        if RDN == _FallbackRDN:
            rdn_params['nf'] = rdn_params.pop('num_feat')
            rdn_params['num_dense_block'] = rdn_params.pop('num_block') # フォールバックはRDB数をnum_dense_blockで受ける

        try:
            self.backbone = RDN(**rdn_params)
        except TypeError as e:
            print(f"[DEBUG] RDN instantiation failed with {rdn_params}: {e}. Trying minimal for fallback.")
            self.backbone = RDN(num_in_ch=1, num_out_ch=1, nf=32, num_dense_block=3, upscale=1)

        # エッジヘッドの入力チャネル数をバックボーンの出力チャネル数に合わせる
        # BasicSRのRDNはnum_featを出力チャネル数とするのが一般的
        # フォールバックRDNはnf
        edge_head_in_channels = rdn_params.get('num_feat', rdn_params.get('nf',32))
        self.edge_head = nn.Conv2d(edge_head_in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_features = self.backbone(x)
        # edge_output = self.edge_head(out_features) # エッジ出力をどう扱うかは後続処理次第
        return out_features # バックボーンの出力をそのまま返す

# ------------------------------------------------------------
# build_model ファクトリ
# ------------------------------------------------------------
def build_model(name: str) -> nn.Module:
    """指定された名前の超解像モデルを構築して返します。

    config.MODEL_CONFIGS から対応するパラメータを読み込んでモデルを初期化します。

    Args:
        name (str): 構築するモデルの名前。config.ALGO_LIST に含まれるべき。

    Returns:
        nn.Module: 構築されたPyTorchモデル。

    Raises:
        ValueError: 指定されたモデル名が config.MODEL_CONFIGS に存在しない場合。
    """
    name_lower = name.lower()
    print(f"[DEBUG] build_model called for: {name_lower}")

    if name_lower not in config.MODEL_CONFIGS:
        raise ValueError(f"Model name '{name}' not found in config.MODEL_CONFIGS. Available: {list(config.MODEL_CONFIGS.keys())}")

    params = config.MODEL_CONFIGS[name_lower].copy() # パラメータをコピーして使用
    params.setdefault('num_in_ch', 1) # デフォルト入力チャネル
    params.setdefault('num_out_ch', 1) # デフォルト出力チャネル
    params.setdefault('upscale', 1) # デフォルトアップスケール (画質改善タスク)

    if name_lower == 'simplesrnet':
        return SimpleSRNet(**params)
    elif 'edgeformer_edsr' in name_lower: # 'edgeformer_edsr' と 'edgeformer_edsr_light'
        return _build_edgeformer_edsr(params)
    elif 'rdn_edge' in name_lower: # 'rdn_edge' と 'rdn_edge_light'
        return _RDN_Edge(params)
    elif 'srmd' in name_lower:
        # SRMDのフォールバック/BasicSR版で引数名が異なる可能性を考慮
        srmd_params = {k: params[k] for k in ['num_in_ch', 'num_out_ch', 'num_feat', 'num_block', 'upscale'] if k in params}
        if SRMD == _FallbackSRMD: # フォールバックの場合の引数名調整
            srmd_params['nf'] = srmd_params.pop('num_feat', 32)
            srmd_params['num_blocks'] = srmd_params.pop('num_block', 6)
        return SRMD(**srmd_params)
    elif 'swinir' in name_lower: # 'swinir' と 'swinir_tiny'
        # SwinIRは引数が多岐にわたるため、paramsをそのまま渡す
        # フォールバックが_FallbackEDSRなので、SwinIR特有の引数は無視される可能性あり
        return SwinIR(**params)
    elif 'nafnet' in name_lower:
        # NAFNetも引数が多岐にわたる
        return NAFNet(**params)
    
    raise ValueError(f'Unknown model name or unhandled model type in build_model: {name}')

