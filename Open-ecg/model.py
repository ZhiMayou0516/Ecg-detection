import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models



# BasicBlock1d from resnet.py
class BasicBlock1d(nn.Module):
    expansion = 1  # 添加这一行，定义 expansion 属性

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualUnit(nn.Module):
    """
    Residual Unit with Preactivation (Identity Mappings in Deep Residual Networks)
    严格复现 Nature Communications 论文 Fig.3 架构
    """

    def __init__(self, n_samples_in, n_samples_out, n_filters_in, n_filters_out,
                 kernel_size=16, downsample=4, dropout_rate=0.2, preactivation=True):
        super(ResidualUnit, self).__init__()
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.downsample = downsample
        self.preactivation = preactivation

        # ★★★ 关键修复：计算正确的 padding，使输出长度 = 输入长度 / downsample ★★★
        # 公式: (L + 2p - k) / s + 1 = L/s  =>  p = (k - s) / 2
        calculated_padding = (kernel_size - downsample) // 2  # (16-4)//2 = 6

        # 主路径第一层 (stride=1)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size,
                               padding=kernel_size // 2, bias=False)  # padding=8，保持长度

        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.post_bn = nn.BatchNorm1d(n_filters_out)
        self.dropout = nn.Dropout(dropout_rate)

        # 主路径第二层 (stride=downsample，进行下采样)
        # 使用计算出的 padding，确保输出长度严格为 n_samples_out
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=calculated_padding, bias=False)

        # Skip connection
        self.skip_conv = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False) \
            if n_filters_in != n_filters_out else None
        self.skip_pool = nn.MaxPool1d(downsample) if downsample > 1 else None

    def forward(self, x):
        identity = x

        # 第一层 (保持长度)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # 第二层 (下采样)
        out = self.conv2(out)

        # Skip connection
        if self.skip_pool is not None:
            identity = self.skip_pool(identity)
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)

        # 相加 (现在尺寸一定匹配)
        out = out + identity

        if self.preactivation:
            out = self.post_bn(out)
            out = F.relu(out)
            out = self.dropout(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, input_channels=12, num_classes=6, kernel_size=16):
        super(ResNet1D, self).__init__()
        self.kernel_size = kernel_size

        # 前面层保持不变...
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.res_unit1 = ResidualUnit(4096, 1024, 64, 128, kernel_size, downsample=4)
        self.res_unit2 = ResidualUnit(1024, 256, 128, 196, kernel_size, downsample=4)
        self.res_unit3 = ResidualUnit(256, 64, 196, 256, kernel_size, downsample=4)
        self.res_unit4 = ResidualUnit(64, 16, 256, 320, kernel_size, downsample=4)

        self.flatten = nn.Flatten()
        # 修改1：fc层输出30维（与ResNet1D_MHL一致）
        self.fc = nn.Linear(320 * 16, 30)
        # 修改2：添加fc2作为分类头
        self.fc2 = nn.Linear(30, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.res_unit4(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc2(x)  # 通过fc2得到logits
        return x

    def get_embedding(self, x):
        """返回30维embedding（用于Stage 2计算prototype）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.res_unit4(x)
        x = self.flatten(x)
        x = self.fc(x)  # 30维
        return x

    def get_embedding_and_logits(self, x):
        """Stage 1专用：同时返回30维embedding和logits"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.res_unit4(x)
        x = self.flatten(x)
        embedding = self.fc(x)  # 30维
        logits = self.fc2(embedding)  # 2k维
        return embedding, logits


def resnet18_model(num_classes=6, **kwargs):
    """兼容原接口，但实际使用论文架构"""
    return ResNet1D(input_channels=12, num_classes=num_classes, kernel_size=16)


class ResNet1D_MHL(nn.Module):
    """
    Stage 2专用模型：去除分类层，仅输出embedding用于多超球面学习
    参考：resnet.py中的ResNet1d_MHL
    """

    def __init__(self, input_channels=12, kernel_size=16):
        super(ResNet1D_MHL, self).__init__()
        self.kernel_size = kernel_size

        # 与ResNet1D相同的前置层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4个Residual Unit
        self.res_unit1 = ResidualUnit(4096, 1024, 64, 128, kernel_size, downsample=4)
        self.res_unit2 = ResidualUnit(1024, 256, 128, 196, kernel_size, downsample=4)
        self.res_unit3 = ResidualUnit(256, 64, 196, 256, kernel_size, downsample=4)
        self.res_unit4 = ResidualUnit(64, 16, 256, 320, kernel_size, downsample=4)

        # Stage 2只需要特征提取，输出320*16=5120维，但参考代码中通过fc压缩到30维
        # 参考resnet.py，添加fc层将5120映射到30维（与参考代码保持一致）
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(320 * 16, 30)  # 30维embedding，与参考代码一致

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.res_unit4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x  # 返回30维embedding用于MHL


def resnet34_MHL(**kwargs):
    """兼容函数，用于Stage 2加载"""
    return ResNet1D_MHL(**kwargs)

# Initialize the model
model = resnet18_model(num_classes=6)  # 6 classes: 1dAVb, RBBB, LBBB, SB, AF, ST
print(model)