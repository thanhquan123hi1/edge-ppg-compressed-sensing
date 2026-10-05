"""
model.py - 1D CNN Hierarchical Upsampling Decoder and Linear Baseline.
Adheres strictly to proposal Section 3.1 and Table 4:
- Linear projection M -> 256, reshaped to [B, 64, 4]
- 4 Hierarchical Upsampling blocks:
  1. Nearest x2 -> Conv1d(64, 64, k=3, p=1) -> BatchNorm1d(64) -> LeakyReLU(0.1) -> [B, 64, 8]
  2. Nearest x4 -> Conv1d(64, 32, k=3, p=1) -> BatchNorm1d(32) -> LeakyReLU(0.1) -> [B, 32, 32]
  3. Nearest x4 -> Conv1d(32, 16, k=3, p=1) -> BatchNorm1d(16) -> LeakyReLU(0.1) -> [B, 16, 128]
  4. Nearest x2 -> Conv1d(16, 8, k=3, p=1) -> BatchNorm1d(8) -> LeakyReLU(0.1) -> [B, 8, 256]
- Output layer: Conv1d(8, 1, k=3, p=1) -> [B, 1, 256] -> squeezed to [B, 256].
- Output is linear (no sigmoid, no clamp).
- Total learnable parameters: 34,049 (M=51), 40,705 (M=77), 53,761 (M=128).
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Upsample + Conv1d + BatchNorm1d + LeakyReLU(0.1)"""
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size=3, padding=1, negative_slope=0.1, use_bn=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

class CNN1DDecoder(nn.Module):
    """
    Hierarchical 1D CNN Decoder for PPG Compressed Sensing reconstruction.
    """
    def __init__(self, M, N=256, use_bn=True):
        super().__init__()
        self.M = M
        self.N = N
        self.use_bn = use_bn
        
        # Projection layer
        self.fc = nn.Linear(M, 256, bias=True)
        
        # 4 Upsampling Blocks
        self.block1 = ConvBlock(64, 64, scale_factor=2, kernel_size=3, padding=1, use_bn=use_bn)
        self.block2 = ConvBlock(64, 32, scale_factor=4, kernel_size=3, padding=1, use_bn=use_bn)
        self.block3 = ConvBlock(32, 16, scale_factor=4, kernel_size=3, padding=1, use_bn=use_bn)
        self.block4 = ConvBlock(16, 8, scale_factor=2, kernel_size=3, padding=1, use_bn=use_bn)
        
        # Linear output layer
        self.out_conv = nn.Conv1d(8, 1, kernel_size=3, padding=1, bias=True)
        
    def forward(self, y):
        """
        y: [B, M] dequantized measurements
        Returns: [B, N] reconstructed PPG window
        """
        # 1. Project and reshape
        x = self.fc(y)                     # [B, 256]
        x = x.view(-1, 64, 4)               # [B, 64, 4]
        
        # 2. Hierarchical upsampling
        x = self.block1(x)                  # [B, 64, 8]
        x = self.block2(x)                  # [B, 32, 32]
        x = self.block3(x)                  # [B, 16, 128]
        x = self.block4(x)                  # [B, 8, 256]
        
        # 3. Output
        x = self.out_conv(x)                # [B, 1, 256]
        return x.squeeze(1)                 # [B, 256]

class LinearDecoder(nn.Module):
    """
    Linear Baseline Decoder: Linear(M, N) mapping y directly to x_hat.
    Used for Table 7 mandatory comparison at M=77.
    """
    def __init__(self, M=77, N=256):
        super().__init__()
        self.fc = nn.Linear(M, N, bias=True)
        
    def forward(self, y):
        return self.fc(y)

class MultiScaleResidualBlock(nn.Module):
    def __init__(self,multiscale=True):
        super().__init__()
        self.branch3=nn.Conv1d(32,16,3,padding=1)
        self.branch7=nn.Conv1d(32,16,7 if multiscale else 3,padding=3 if multiscale else 1)
        self.fuse=nn.Conv1d(32,32,3,padding=1)
        self.act=nn.LeakyReLU(.1)

    def forward(self,x):
        features=torch.cat([self.act(self.branch3(x)),self.act(self.branch7(x))],dim=1)
        return self.act(x+self.fuse(features))

class MSResCNN1DDecoder(nn.Module):
    """Full-resolution linear reconstruction plus trainable multi-scale residual."""
    def __init__(self,M,N=256,global_skip=True,multiscale=True):
        super().__init__()
        if N != 256:
            raise ValueError('Locked architecture requires N=256')
        self.M=M;self.N=N;self.global_skip=global_skip
        self.fc=nn.Linear(M,N)
        self.stem=nn.Conv1d(1,32,5,padding=2)
        self.act=nn.LeakyReLU(.1)
        self.blocks=nn.ModuleList([MultiScaleResidualBlock(multiscale) for _ in range(3)])
        self.head=nn.Conv1d(32,1,5,padding=2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,y):
        x0=self.fc(y)
        f=self.act(self.stem(x0.unsqueeze(1)))
        for block in self.blocks:
            f=block(f)
        residual=self.head(f).squeeze(1)
        return x0+residual if self.global_skip else residual

def build_decoder(method,M):
    if method=='cnn':return CNN1DDecoder(M)
    if method=='linear':return LinearDecoder(M)
    if method=='msres':return MSResCNN1DDecoder(M)
    if method=='msres_noskip':return MSResCNN1DDecoder(M,global_skip=False)
    if method=='msres_single':return MSResCNN1DDecoder(M,multiscale=False)
    raise ValueError(f'Unknown model {method}')

def count_parameters(model):
    """Counts total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Verifying CNN 1D Architecture and Parameter Counts...")
    for M, expected_params in [(51, 34049), (77, 40705), (128, 53761)]:
        model = CNN1DDecoder(M)
        n_params = count_parameters(model)
        dummy_in = torch.randn(4, M)
        dummy_out = model(dummy_in)
        print(f"M={M}: Output shape={dummy_out.shape}, Params={n_params} (Expected: {expected_params})")
        assert n_params == expected_params, f"Parameter mismatch for M={M}: {n_params} != {expected_params}"
        assert dummy_out.shape == (4, 256), f"Shape mismatch: {dummy_out.shape}"
        
    lin_model = LinearDecoder(77, 256)
    print(f"Linear(77, 256) params: {count_parameters(lin_model)}")
    print("All architecture checks passed.")
