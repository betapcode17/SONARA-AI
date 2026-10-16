import torch
import torch.nn as nn
import torch.nn.functional as F


class IBN(nn.Module):
    """Instance-Batch Normalization (IBN) kết hợp IN (nửa số kênh) và BN (nửa còn lại)."""
    def __init__(self, planes: int):
        super(IBN, self).__init__()
        half1 = planes // 2
        self.half = half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(planes - half1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, self.half, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), dim=1)


class GeMPooling(nn.Module):
    """Generalized Mean (GeM) Pooling với tham số khả vi p (mặc định p=3.0 theo ByteCover / ByteHum)."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamp = x.clamp(min=self.eps).pow(self.p)
        pooled = F.adaptive_avg_pool2d(x_clamp, (1, 1)).pow(1.0 / self.p)
        return pooled.flatten(1)


class BasicBlockIBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None, use_ibn: bool = True):
        super(BasicBlockIBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = IBN(planes) if use_ibn else nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetIBN(nn.Module):
    """Mạng trích xuất đặc trưng ResNet-IBN-18 chuyên dụng cho bài toán Cover Song Identification."""
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 512,
        gem_p: float = 3.0,
        dropout: float = 0.1,
        pretrained: bool = False
    ):
        super(ResNetIBN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockIBN, 64, 2, stride=1, use_ibn=True)
        self.layer2 = self._make_layer(BasicBlockIBN, 128, 2, stride=2, use_ibn=True)
        self.layer3 = self._make_layer(BasicBlockIBN, 256, 2, stride=2, use_ibn=True)
        self.layer4 = self._make_layer(BasicBlockIBN, 512, 2, stride=2, use_ibn=False)

        self.gem_pool = GeMPooling(p=gem_p)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512, embedding_dim)

        # Khởi tạo ma trận trực giao (Orthogonal Initialization) - Tương đương PCA-FC
        nn.init.orthogonal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        if pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1, use_ibn=True):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample, use_ibn=use_ibn)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ibn=use_ibn))
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        try:
            import torchvision.models as models
            tv_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            tv_state = tv_model.state_dict()
            if 'conv1.weight' in tv_state:
                self.conv1.weight.data = tv_state['conv1.weight'].mean(dim=1, keepdim=True)
            if 'bn1.weight' in tv_state:
                self.bn1.weight.data.copy_(tv_state['bn1.weight'])
                self.bn1.bias.data.copy_(tv_state['bn1.bias'])
                self.bn1.running_mean.copy_(tv_state['bn1.running_mean'])
                self.bn1.running_var.copy_(tv_state['bn1.running_var'])
            print("[*] Đã nạp thành công trọng số ImageNet Pretrained cho ResNet-IBN!")
        except Exception as e:
            print(f"[*] Chế độ offline / Sử dụng Kaiming initialization: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 5: # (B, N, C, F, T)
            B, N, C, F_dim, T_dim = x.shape
            x = x.view(B * N, C, F_dim, T_dim)
            is_seq = True
        else:
            is_seq = False

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        feat = self.dropout(self.gem_pool(x))
        emb = F.normalize(self.fc(feat), p=2, dim=1)

        if is_seq:
            emb = emb.view(B, N, -1)
        return emb
