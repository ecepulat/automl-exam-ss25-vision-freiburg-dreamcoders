import torch.nn as nn
import torch.nn.functional as F

# This Block class defines a single unit of the architecture based on sampled hyperparameters.
# It supports depthwise or standard convolutions, optional residual connections, and squeeze-and-excitation blocks.
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_residual = config["use_residual"]
        self.downsample = config["downsample"]
        self.use_depthwise = config["use_depthwise"]

        stride = 2 if self.downsample else 1
        in_channels = config["filters"]
        out_channels = config["filters"]

        if self.use_depthwise:
            # Depthwise separable convolution block
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=config["kernel"], stride=stride,
                          padding=config["kernel"] // 2, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # Standard convolution block
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=config["kernel"], stride=stride,
                          padding=config["kernel"] // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.use_se = config["use_se"]
        if self.use_se:
            # Squeeze-and-Excitation module
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            scale = self.se(out)
            out = out * scale
        if self.use_residual and out.shape == x.shape:
            out = out + x
        return out


# CustomNet builds a full model using a sequence of sampled blocks, pooling, and classification layers.
class CustomNet(nn.Module):
    def __init__(self, blocks, dropout, pool_type, num_classes):
        super().__init__()
        # Stem layer to map from 3 input channels to first block's filter count
        self.stem = nn.Sequential(
            nn.Conv2d(3, blocks[0]["filters"], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(blocks[0]["filters"]),
            nn.ReLU(inplace=True)
        )

        # Stack all block configurations
        layers = []
        for block_cfg in blocks:
            layers.append(Block(block_cfg))
        self.layers = nn.Sequential(*layers)

        # Global pooling layer choice
        self.pool_type = pool_type
        if pool_type == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.global_pool = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(blocks[-1]["filters"], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


# Entry point function to be called from Optuna pipeline
# Takes sampled hyperparameters (blocks, dropout, pooling) and returns a PyTorch model

def build_model_from_config(blocks, dropout, pool_type, num_classes):
    return CustomNet(blocks, dropout, pool_type, num_classes)
