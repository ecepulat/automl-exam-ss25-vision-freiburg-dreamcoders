import torch
import torch.nn as nn
import torch.nn.functional as F

# Single convolutional block supporting depthwise, SE, and residual connection
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_residual = config["use_residual"]
        self.downsample = config["downsample"]
        self.use_depthwise = config["use_depthwise"]

        stride = 2 if self.downsample else 1
        in_channels = config["in_channels"]
        out_channels = config["filters"]

        if self.use_depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=config["kernel"], stride=stride,
                          padding=config["kernel"] // 2, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=config["kernel"], stride=stride,
                          padding=config["kernel"] // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.use_se = config["use_se"]
        if self.use_se:
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

# Full model constructed from blocks + global pooling + classifier
class CustomNet(nn.Module):
    def __init__(self, blocks, dropout, pool_type, num_classes, input_resolution=(128, 128)):
        super().__init__()

        # Initial stem conv layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, blocks[0]["filters"], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(blocks[0]["filters"]),
            nn.ReLU(inplace=True)
        )

        # Stack of sampled blocks
        layers = []
        in_channels = blocks[0]["filters"]
        for block_cfg in blocks:
            block_cfg["in_channels"] = in_channels
            layers.append(Block(block_cfg))
            in_channels = block_cfg["filters"]
        self.layers = nn.Sequential(*layers)

        # Pooling strategy
        if pool_type == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.global_pool = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # Dynamically compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_resolution)
            x = self.stem(dummy)
            x = self.layers(x)
            x = self.global_pool(x)
            x = x.view(1, -1)
            feature_dim = x.shape[1]

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

# External entry point to create the model in search pipeline
def build_model_from_config(blocks, dropout, pool_type, num_classes, input_resolution=(128, 128)):
    return CustomNet(blocks, dropout, pool_type, num_classes, input_resolution)
