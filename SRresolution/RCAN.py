import torch
import torch.nn as nn
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """標準的な畳み込み層を返す"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class CALayer(nn.Module):
    """Channel Attention (CA) Layer"""
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature Channel Downscale and Upscale
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB)"""
    def __init__(
        self, conv, n_feat, kernel_size, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True)):
        
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup(nn.Module):
    """Residual Group (RG)"""
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)
            ) for _ in range(n_resblocks)]
        
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RCAN(nn.Module):
    """Residual Channel Attention Network (RCAN)"""
    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64, scale=2, n_colors=1, conv=default_conv):
        super(RCAN, self).__init__()
        
        kernel_size = 3
        reduction = 16
        
        # 1. Shallow Feature Extraction
        self.head = conv(n_colors, n_feats, kernel_size)
        
        # 2. Deep Feature Extraction (Residual in Residual structure)
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks
            ) for _ in range(n_resgroups)]
        
        self.body = nn.Sequential(*modules_body)
        self.body_conv = conv(n_feats, n_feats, kernel_size)

        # 3. Upsampling
        modules_tail = [
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size, padding=(kernel_size//2)),
            nn.PixelShuffle(scale),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # Shallow Feature
        x = self.head(x)
        
        # Deep Feature (RIR)
        res = self.body(x)
        res = self.body_conv(res)
        res += x
        
        # Upsampling
        x = self.tail(res)
        
        return x

# --- 使用例 ---
if __name__ == '__main__':
    # モデルのパラメータ
    # ご自身のタスクに合わせて調整してください
    num_res_groups = 5     # ResidualGroupの数 (論文では10)
    num_res_blocks = 10    # 各ResidualGroup内のRCABの数 (論文では20)
    num_features = 64      # 特徴マップの数
    scale_factor = 1       # 超解像の倍率。今回は高画質化なので1
    num_colors = 1         # 入力画像のチャンネル数（白黒なので1）

    # モデルのインスタンス化
    model = RCAN(
        n_resgroups=num_res_groups,
        n_resblocks=num_res_blocks,
        n_feats=num_features,
        scale=scale_factor,
        n_colors=num_colors
    )
    
    print(f"RCANモデルの総パラメータ数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ダミーの入力データを作成
    # (batch_size, channels, height, width)
    # 縦800, 横256のCD-SEM画像を想定
    dummy_input = torch.randn(4, 1, 800, 256)

    # モデルで推論を実行
    output = model(dummy_input)

    # 出力サイズの確認
    print(f"入力テンソルの形状: {dummy_input.shape}")
    print(f"出力テンソルの形状: {output.shape}")

    # scale=1 の場合、出力サイズは入力サイズと同じになる
    assert dummy_input.shape == output.shape