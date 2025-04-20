# models/generator.py
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """正弦时间步嵌入，用于给扩散模型输入t时刻信息。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B], integer time steps
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb  # [B, dim]

class DoubleConv(nn.Module):
    """
    一个简单的2次卷积块: conv->ReLU->conv->ReLU
    常见于U-Net编码/解码阶段
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class DownSample(nn.Module):
    """下采样: MaxPool2d + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    """上采样: ConvTranspose2d + 拼接skip + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up_transpose = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch*2, out_ch)
    def forward(self, x, skip):
        x = self.up_transpose(x)
        # 与 skip 拼接
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = nn.functional.pad(x, [0, diffX, 0, diffY])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ConditionalUNet(nn.Module):
    """
    简易条件U-Net, 输入[B,3,H,W], 输出[B,3,H,W].
    在中间层加入时间+类别嵌入, 用于扩散模型.
    """
    def __init__(self, img_channels=3, class_count=150, base_dim=64, time_emb_dim=256):
        super().__init__()
        self.class_count = class_count

        # 时间/类别嵌入
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.class_emb = nn.Embedding(class_count, time_emb_dim)

        # encoder
        self.conv_in = DoubleConv(img_channels, base_dim)      # 3->64
        self.down1   = DownSample(base_dim, base_dim*2)        # 64->128
        self.down2   = DownSample(base_dim*2, base_dim*4)      # 128->256

        # mid
        self.mid = DoubleConv(base_dim*4, base_dim*4)

        # decoder
        self.up1 = UpSample(base_dim*4, base_dim*2)            # 256->128
        self.up2 = UpSample(base_dim*2, base_dim)              # 128->64

        # 最终输出
        self.conv_out = nn.Conv2d(base_dim, img_channels, kernel_size=3, padding=1)

        # 用于将 embedding注入中间层
        self.emb_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, base_dim*4),
            nn.ReLU()
        )

    def forward(self, x, t, class_labels):
        """
        x: [B,3,H,W], t: [B], class_labels: [B]
        return: 预测噪声 [B,3,H,W]
        """
        # 1) 计算时间+类别嵌入
        emb_t = self.time_emb(t)     # [B, time_emb_dim]
        emb_t = self.time_mlp(emb_t) # [B, time_emb_dim]
        if class_labels is not None:
            emb_c = self.class_emb(class_labels)  # [B, time_emb_dim]
        else:
            emb_c = torch.zeros_like(emb_t)
        emb = emb_t + emb_c
        emb = self.emb_mlp(emb)  # [B, base_dim*4]

        # 2) encoder
        c1 = self.conv_in(x)         # [B,64,H,W]
        c2 = self.down1(c1)         # [B,128,H/2,W/2]
        c3 = self.down2(c2)         # [B,256,H/4,W/4]

        # 3) mid
        mid = self.mid(c3)          # [B,256,H/4,W/4]
        B, C, H4, W4 = mid.shape
        emb_4d = emb.view(B, -1, 1, 1).expand(B, -1, H4, W4)
        mid = mid + emb_4d          # 在中间层加入embedding

        # 4) decoder
        u1 = self.up1(mid, c2)      # [B,128,H/2,W/2]
        u2 = self.up2(u1, c1)       # [B,64,H,W]

        out = self.conv_out(u2)     # [B,3,H,W]
        return out

class DiffusionGenerator:
    """
    扩散过程封装:
      - noise_image: 前向加噪
      - train_step: 单步训练(预测噪声)
      - sample: 逆扩散生成
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        # 线性beta调度
        self.T = 1000
        self.betas = torch.linspace(1e-4, 2e-2, self.T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cum = torch.cumprod(self.alphas, dim=0)
        self.img_size = 64  # 给采样用

    def sample_noise(self, shape):
        return torch.randn(shape, device=self.device)

    def noise_image(self, x0, t):
        """
        x0: [B,3,H,W]
        t: [B], 各图随机时刻
        返回 (x_t, noise)
        """
        B, C, H, W = x0.shape
        a = self.alphas_cum[t]  # shape=[B]
        a = a.view(B,1,1,1)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(a)*x0 + torch.sqrt(1.0 - a)*noise
        return x_t, noise

    def train_step(self, x0, class_labels, optimizer, criterion):
        """
        单步训练:
          1) 随机采样t
          2) 前向加噪
          3) 模型预测噪声, 与真实noise对比
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        x_t, noise = self.noise_image(x0, t)
        pred_noise = self.model(x_t, t, class_labels)

        loss = criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def sample(self, class_label=None):
        """
        从纯噪声逆向生成一张图片
        """
        self.model.eval()
        with torch.no_grad():
            x = self.sample_noise((1,3,self.img_size,self.img_size))
            for i in reversed(range(self.T)):
                t = torch.tensor([i], device=self.device)
                label_input = None
                if class_label is not None:
                    label_input = torch.tensor([class_label], device=self.device)
                pred_noise = self.model(x, t, label_input)

                alpha_t = self.alphas[i]
                alpha_cum_t = self.alphas_cum[i]
                if i > 0:
                    sigma_t = torch.sqrt((1 - alpha_t) * (1 - alpha_cum_t) / alpha_t)
                else:
                    sigma_t = 0
                x = (1/torch.sqrt(alpha_t)) * (x - (1 - alpha_t)/torch.sqrt(1-alpha_cum_t)*pred_noise)
                if i>0:
                    x = x + sigma_t*torch.randn_like(x)

            img = x.clamp(-1,1)  # 最终输出[-1,1]
            return img[0]  # shape=[3,H,W]
