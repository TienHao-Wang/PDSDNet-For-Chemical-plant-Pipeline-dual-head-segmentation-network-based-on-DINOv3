import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d # 处理斜向管线

# --- 改进 A：条带卷积 (StripConv) - 保留 H+V 几何先验 ---
class StripConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15):
        super().__init__()
        p = kernel_size // 2
        self.conv_h = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, p))
        self.conv_v = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(p, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv_h(x) + self.conv_v(x)))

# --- 改进 B：条带池化 (StripPooling) - 保留全局线性上下文 ---
class StripPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = F.interpolate(self.pool_h(x), (h, w), mode='bilinear', align_corners=True)
        x_v = F.interpolate(self.pool_v(x), (h, w), mode='bilinear', align_corners=True)
        return torch.sigmoid(self.conv1x1(x_h + x_v)) * x

# --- 改进 C：坐标注意力 (CA) - 解决位置感知与线性定位 ---
class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))

# --- 改进 D：可变形条带卷积 (DCN) - 核心：解决斜向与弯曲管线 ---
class DeformableStripConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(in_ch, kernel_size * kernel_size, kernel_size=3, padding=1)
        self.conv = DeformConv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.relu(self.bn(self.conv(x, offset, mask=mask)))

# --- 级联细化头 (CRM) - 核心：修复断裂与平滑边缘 ---
class RefinementHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1)
        )

    def forward(self, feat, mask_coarse):
        x = torch.cat([feat, torch.sigmoid(mask_coarse)], dim=1)
        return mask_coarse + self.refine(x)

class LDoffset(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15):
        super().__init__()
        p = kernel_size // 2
        self.conv_h = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, p))
        self.conv_v = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(p, 0))


    def forward(self, x):
        return (self.conv_h(x) + self.conv_v(x))
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, dckernel_size=3, lkernel_size=15):
        super().__init__()
        self.offset_conv = LDoffset(in_ch=in_ch, out_ch=2*dckernel_size*dckernel_size, kernel_size=lkernel_size)
        #self.offset_conv = nn.Conv2d(in_ch, 2 * dckernel_size * dckernel_size, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(in_ch, dckernel_size * dckernel_size, kernel_size=3, padding=1)
        self.conv = DeformConv2d(in_ch, out_ch, kernel_size=dckernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.relu(self.bn(self.conv(x, offset, mask=mask)))

class PipeDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DSConv(out_ch, out_ch)
            #CoordinateAttention(out_ch, out_ch), # 增强位置感知
            #DeformableStripConv(out_ch, out_ch),  # 增强斜向适应
            #StripConv(out_ch, out_ch)            # 增强几何形态
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return self.conv_refine(torch.cat([x, skip], dim=1))


class DINOPipeNet_RS(nn.Module):
    def __init__(self, dinov3_path, embed_dim=1024):
        super().__init__()
        # 1. 加载 DINOv3 骨干
        self.backbone = torch.hub.load(r"./dinov3/", 'dinov3_vitl16', source='local', weights=dinov3_path)
        for p in self.backbone.parameters(): p.requires_grad = False

        # 2. Neck：SimpleFPN + StripPooling
        self.neck_p5 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(embed_dim, 256, 1))
        self.neck_p4 = nn.Conv2d(embed_dim, 256, 1)
        self.neck_p3 = nn.Sequential(nn.ConvTranspose2d(embed_dim, 256, 2, 2), nn.BatchNorm2d(256), nn.ReLU())
        self.neck_p2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, 256, 4, 4), nn.BatchNorm2d(256), nn.ReLU())

        # 核心：在 Neck 加入条带池化
        self.sp_module = StripPooling(256)

        # 3. 逐级解码 (集成三大改进模块)
        self.dec4 = PipeDecoderBlock(256, 256, 128)
        self.dec3 = PipeDecoderBlock(128, 256, 64)
        self.dec2 = PipeDecoderBlock(64, 256, 32)

        # 4. 边缘预测模块 (独立分支)
        self.edge_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # 5. 级联细化路径 (1/4 -> 1/1)
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4)
        self.mask_coarse_head = nn.Conv2d(16, 1, kernel_size=1)
        self.crm = RefinementHead(16)

    def forward(self, x):
        # A. 多层特征提取
        with torch.no_grad():
            layers = self.backbone.get_intermediate_layers(x, n=[7, 15, 23], reshape=True)
        f_shallow, f_mid, f_deep = layers

        # B. Neck 构建与 StripPooling 增强
        p5 = self.sp_module(self.neck_p5(f_deep))
        p4 = self.neck_p4(f_deep)
        p3 = self.neck_p3(f_mid)
        p2 = self.neck_p2(f_shallow)

        # C. 逐级上采样解码
        d4 = self.dec4(p5, p4)
        d3 = self.dec3(d4, p3)
        d2 = self.dec2(d3, p2)  # 1/4 分辨率, 32通道

        # D. 边缘分支预测 (用于辅助 Loss)
        edge_pred = self.edge_head(d2)

        # E. 级联细化推理 (CRM)
        f_high = self.final_up(d2)  # 上采样至原图大小 (16通道)
        mask_coarse = self.mask_coarse_head(f_high)
        mask_final = self.crm(f_high, mask_coarse)

        return mask_final, edge_pred


def build_model(dinov3_path):
    return DINOPipeNet_RS(dinov3_path=dinov3_path)
