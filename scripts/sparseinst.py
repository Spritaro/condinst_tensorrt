import math

import torch
import torch.nn.functional as F
from torch import nn

import torchvision


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()

        def conv1x1_bn_relu(in_channels, out_channels):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
        self.convs = nn.ModuleList([conv1x1_bn_relu(in_channels, channels) for i in range(4)])
        self.out_conv = conv1x1_bn_relu(in_channels+channels*4, in_channels)
        return

    def forward(self, in_feature):
        """
        Params:
            in_feature: Tensor[batch, D, H, W]
        Returns:
            out_feature: Tensor[batch, D, H, W]
        """
        batch, D, H, W = in_feature.shape

        # NOTE: TensorRT7 does not support F.adaptive_avg_pool2d
        def adaptive_avg_pool2d(feature, output_size):
            stride = (H//output_size, W//output_size)
            kernel_size = (H-(output_size-1)*stride[0], W-(output_size-1)*stride[1])
            feature = F.avg_pool2d(feature, kernel_size=kernel_size, stride=kernel_size, padding=0)
            return feature
        output_sizes = [1, 2, 3, 6]
        xs = [in_feature]
        for output_size, conv in zip(output_sizes, self.convs):
            x = adaptive_avg_pool2d(in_feature, output_size)
            x = conv(x)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            xs.append(x)

        x = torch.cat(xs, dim=1)
        out_feature = self.out_conv(x)
        return out_feature


class Encoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        def conv1x1_bn(in_channels, out_channels):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)
        self.lateral_conv3 = conv1x1_bn(512, num_channels)
        self.lateral_conv4 = conv1x1_bn(1024, num_channels)
        self.lateral_conv5 = conv1x1_bn(2048, num_channels)

        self.ppm = PyramidPoolingModule(num_channels, num_channels//4)

        def conv3x3_bn(in_channels, out_channels):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)
        self.conv3 = conv3x3_bn(num_channels, num_channels)
        self.conv4 = conv3x3_bn(num_channels, num_channels)
        self.conv5 = conv3x3_bn(num_channels, num_channels)

        self.encoder_projection = conv1x1_bn(num_channels * 3, num_channels)

        def initialize_encoder(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(initialize_encoder)
        return

    def forward(self, c3, c4, c5):
        # FPN
        l5 = self.lateral_conv5(c5)
        l4 = self.lateral_conv4(c4)
        l3 = self.lateral_conv3(c3)
        p5 = self.ppm(l5) # Pyramid Pooling Module
        p4 = l4 + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = l3 + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)

        # 3x3 convs
        x5 = self.conv5(p5)
        x4 = self.conv4(p4)
        x3 = self.conv3(p3)

        # Concat
        x5 = F.interpolate(x5, scale_factor=4, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x3, x4, x5], dim=1)
        feature = self.encoder_projection(x)
        return feature


class Decoder(nn.Module):
    def __init__(self, num_classes, num_instances, num_channels, num_kernel_channels):
        super().__init__()

        def stack_conv3x3_bn_relu(in_channels, out_channels, num_stack):
            layers = []
            for i in range(num_stack):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            return nn.Sequential(*layers)
        self.inst_branch = stack_conv3x3_bn_relu(num_channels+2, num_channels, num_stack=4)
        self.mask_branch = stack_conv3x3_bn_relu(num_channels+2, num_channels, num_stack=4)
        self.mask_projection = nn.Conv2d(num_channels, num_kernel_channels, kernel_size=1, padding=0, bias=True)

        self.f_iam = nn.Conv2d(num_channels, num_instances, kernel_size=3, padding=1, bias=True)

        self.class_head = nn.Linear(num_channels, num_classes)
        self.score_head = nn.Linear(num_channels, 1)
        self.kernel_head = nn.Linear(num_channels, num_kernel_channels)

        # Initialize
        def initialize(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.inst_branch.apply(initialize)
        self.mask_branch.apply(initialize)
        self.mask_projection.apply(initialize)
        self.f_iam.apply(initialize)
        self.class_head.apply(initialize)
        self.score_head.apply(initialize)
        self.kernel_head.apply(initialize)

        # Initialize head bias
        # NOTE: see Focal Loss paper for detail https://arxiv.org/abs/1708.02002
        pi = 0.01
        bias = -math.log((1 - pi) / pi)
        nn.init.constant_(self.f_iam.bias, bias)
        nn.init.constant_(self.class_head.bias, bias)
        nn.init.constant_(self.score_head.bias, bias)
        return

    def forward(self, feature):
        """
        Params:
            feature: Tensor[batch, D, H, W]
        Returns:
            class_logits: Tensor[batch, N, C]
            mask_logits: Tensor[batch, N, H, W]
        """
        # Instance branch
        inst_feature = self.inst_branch(feature)
        batch, D, H, W = inst_feature.shape

        # Instance activation map
        iam = self.f_iam(inst_feature) # [batch, N, H, W]
        iam = iam.sigmoid()
        iam = iam.view(batch, -1, H*W) # [batch, N, (H*W)]
        iam = iam / (iam.sum(dim=2, keepdim=True) + 1e-6) # Normalize

        # Instance aware feature
        inst_feature = inst_feature.view(batch, D, -1) # [batch, D, (H*W)]
        inst_feature = torch.transpose(inst_feature, 1, 2) # [batch, (H*W), D]
        inst_aware_feature = torch.matmul(iam, inst_feature) # [batch, N, D] = [batch, N, (H*W)] * [batch, (H*W), D]

        # Heads
        class_logits = self.class_head(inst_aware_feature) # [batch, N, C]
        score_logits = self.score_head(inst_aware_feature) # [batch, N, 1]
        kernel_logits = self.kernel_head(inst_aware_feature) # [batch, N, kernel]

        # Mask branch
        mask_feature = self.mask_branch(feature)
        mask_feature = self.mask_projection(mask_feature) # [batch, kernel, H, W]
        mask_logits = self.generate_mask(kernel_logits, mask_feature) # [batch, N, H, W]

        iam = iam.view(batch, -1, H, W)
        return class_logits, score_logits, mask_logits, iam

    def generate_mask(self, kernel_logits, mask_feature):
        """
        Params:
            kernel_logits: Tensor[batch, N, kernel]
            mask_feature: Tensor[batch, kernel, H, W]
        Returns:
            mask_preds: Tensor[batch, N, H, W]
        """
        batch, N, kernel = kernel_logits.shape
        _, _, H, W = mask_feature.shape

        # [batch, N, (H*W)] = [batch, N, kernel] * [batch, kernel, (H*W)]
        mask_logits = torch.matmul(kernel_logits, mask_feature.view(batch, kernel, -1))
        mask_logits = mask_logits.view(batch, -1, H, W) # Tensor[batch, N, H, W]

        return mask_logits


class SparseInst(nn.Module):
    def __init__(self, mode, input_channels, num_classes, num_instances):
        super().__init__()
        assert mode in ['training', 'inference']
        self.mode = mode
        num_channels = 256
        num_kernel_channels = 128

        self.backbone = torchvision.models.resnet50(pretrained=True)

        self.encoder = Encoder(num_channels)
        self.decoder = Decoder(num_classes, num_instances, num_channels, num_kernel_channels)

        # def freeze_bn(m):
        #     if isinstance(m, nn.BatchNorm2d):
        #         # m.track_running_stats = False
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False
        #         # m.eval()
        # self.backbone.apply(freeze_bn)

        # Change number of input channels
        if input_channels != 3:
            output_channels, _, h, w = self.backbone.conv1.weight.shape
            weight = torch.zeros(output_channels, input_channels, h, w)
            nn.init.normal_(weight, std=0.01)
            weight[:, :3, :, :] = self.backbone.conv1.weight
            self.backbone.conv1.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, images):
        # Convert input images to FP32 or FP16 depending on backbone dtype
        images = images.to(dtype=self.backbone.conv1.weight.dtype)

        # Backbone
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)  # 1/4
        c3 = self.backbone.layer2(c2)  # 1/8
        c4 = self.backbone.layer3(c3)  # 1/16
        c5 = self.backbone.layer4(c4)  # 1/32

        feature = self.encoder(c3, c4, c5) # 1/8
        feature = self.add_coordinate(feature)
        class_logits, score_logits, mask_logits, iam = self.decoder(feature)

        if self.mode == 'training':
            return class_logits, score_logits, mask_logits, iam
        else:
            class_preds = torch.sigmoid(class_logits) # classification predictions
            score_preds = torch.sigmoid(score_logits) # objectness score predictions
            scores = torch.sqrt(class_preds * score_preds) # [batch, N, C]
            # scores = class_preds
            scores, labels = torch.max(scores, dim=2)
            return labels.int(), scores.float(), mask_logits.float()

    def add_coordinate(self, feature):
        batch, D, H, W = feature.shape
        dtype = feature.dtype
        device = feature.device

        # NOTE: TensorRT7 does not support INT32 types for the NEG operator.
        # NOTE: TensorRT7 does not support float range operation. Use cast instead.
        coord_x = torch.arange(0, W, 1, dtype=torch.int32, device=device) # [W]
        coord_y = torch.arange(0, H, 1, dtype=torch.int32, device=device) # [H]
        coord_x = coord_x.to(dtype) / W
        coord_y = coord_y.to(dtype) / H
        coord_y, coord_x = torch.meshgrid(coord_y, coord_x) # [H, W], [H, W]

        coord_x = coord_x.view(1, 1, H, W).expand(batch, 1, H, W)
        coord_y = coord_y.view(1, 1, H, W).expand(batch, 1, H, W)
        feature = torch.cat([feature, coord_x, coord_y], dim=1)
        return feature
