import math
import numpy as np
import scipy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.ops.focal_loss import sigmoid_focal_loss

from loss import dice_score_matrix, dice_loss_vector


# class PyramidPoolingModule(nn.Module):
#     def __init__(self, in_channels, channels):
#         super().__init__()

#         self.convs = nn.ModuleList(
#             [nn.Sequential(
#                 nn.Conv2d(in_channels, channels, kernel_size=1, padding=0, bias=True),
#                 nn.ReLU()) for i in range(4)])
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels + channels * 4, in_channels, kernel_size=1, padding=0, bias=True),
#             nn.ReLU())
#         return

#     def forward(self, in_feature):
#         """
#         Params:
#             in_feature: Tensor[batch, D, H, W]
#         Returns:
#             out_feature: Tensor[batch, D, H, W]
#         """
#         batch, D, H, W = in_feature.shape

#         divs = [1, 2, 4, 8]
#         xs = [in_feature]
#         for div, conv in zip(divs, self.convs):
#             # NOTE: TensorRT7 does not support AdaptiveAvgPool2d
#             kernel_size = (H//div, W//div)
#             x = F.avg_pool2d(in_feature, kernel_size=kernel_size, stride=kernel_size, padding=0)
#             x = conv(x)
#             x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#             xs.append(x)

#         x = torch.cat(xs, dim=1)
#         out_feature = self.out_conv(x)
#         return out_feature


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

        # self.ppm = PyramidPoolingModule(num_channels, num_channels//4)

        def conv3x3_bn(in_channels, out_channels):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)
        self.conv3 = conv3x3_bn(num_channels, num_channels)
        self.conv4 = conv3x3_bn(num_channels, num_channels)
        self.conv5 = conv3x3_bn(num_channels, num_channels)

        self.encoder_projection = conv1x1_bn(num_channels * 3, num_channels)

        def initialize(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(initialize)
        return

    def forward(self, c3, c4, c5):
        # TODO: add Pyarmid Pooling Module

        # FPN
        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.lateral_conv3(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)

        # # Pyramid Pooling Module
        # p5 = self.ppm(p5)

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
                layers.append(nn.ReLU())
                in_channels = out_channels
            return nn.Sequential(*layers)
        self.inst_branch = stack_conv3x3_bn_relu(num_channels+2, num_channels, num_stack=4)
        self.mask_branch = stack_conv3x3_bn_relu(num_channels+2, num_channels, num_stack=4)
        self.mask_projection = nn.Conv2d(num_channels, num_kernel_channels, kernel_size=1, padding=0, bias=True)

        self.f_iam = nn.Sequential(
            nn.Conv2d(num_channels, num_instances, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid())

        self.class_head = nn.Linear(num_channels, num_classes)
        self.score_head = nn.Linear(num_channels, 1)
        self.kernel_head = nn.Linear(num_channels, num_kernel_channels)

        # Initialize
        def initialize_branch(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.inst_branch.apply(initialize_branch)
        self.mask_branch.apply(initialize_branch)
        self.mask_projection.apply(initialize_branch)

        def initialize_head(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.f_iam.apply(initialize_head)
        self.class_head.apply(initialize_head)
        self.score_head.apply(initialize_head)
        self.kernel_head.apply(initialize_head)

        # Initialize last layer of F-iam and class head
        # NOTE: see Focal Loss paper for detail https://arxiv.org/abs/1708.02002
        pi = 0.01
        bias = -math.log((1 - pi) / pi)
        nn.init.constant_(self.f_iam[0].bias, bias)
        nn.init.constant_(self.class_head.bias, bias)
        return

    def forward(self, feature):
        """
        Params:
            feature: Tensor[batch, D*3, H, W]
        Returns:
            class_logits: Tensor[batch, N, C]
            mask_logits: Tensor[batch, N, H, W]
        """
        # Instance branch
        inst_feature = self.inst_branch(feature)
        batch, D, H, W = inst_feature.shape

        # Instance activation map
        iam = self.f_iam(inst_feature) # [batch, N, H, W]
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
        mask_logits = self.generate_mask(kernel_logits, mask_feature) # [batch, 1, H, W]

        return class_logits, score_logits, mask_logits

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
        #         assert(hasattr(m, 'track_running_stats'))
        #         m.track_running_stats = False
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
        class_logits, score_logits, mask_logits = self.decoder(feature)

        if self.mode == 'training':
            return class_logits, score_logits, mask_logits
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

        # NOTE: TensorRT7 does not support float range operation. Use cast instead.
        coord_x = torch.arange(-W, W, 2, dtype=torch.int32, device=device) # [W]
        coord_y = torch.arange(-H, H, 2, dtype=torch.int32, device=device) # [H]
        coord_x = coord_x.to(dtype) / W
        coord_y = coord_y.to(dtype) / H
        coord_y, coord_x = torch.meshgrid(coord_y, coord_x) # [H, W], [H, W]

        coord_x = coord_x.view(1, 1, H, W).expand(batch, 1, H, W)
        coord_y = coord_y.view(1, 1, H, W).expand(batch, 1, H, W)
        feature = torch.cat([feature, coord_x, coord_y], dim=1)
        return feature

    def loss(self, class_logits, score_logits, mask_logits, targets):
        """
        Params:
            class_logits: Tensor[batch, N, C]
            score_logits: Tensor[batch, N, 1]
            mask_logits: Tensor[batch, N, H, W]
            targets: List[List[Dict{'class_labels': int, 'segmentation': ndarray[imageH, imageW]}]]
        Returns:
            class_loss: Tensor[]
            score_loss: Tensor[]
            mask_loss: Tensor[]
        """
        batch, N, C = class_logits.shape
        _, _, H, W = mask_logits.shape
        dtype = class_logits.dtype
        device = class_logits.device

        # Upsample masks to 1/4 of input size
        mask_logits = F.interpolate(mask_logits, scale_factor=2, mode='bilinear', align_corners=False) # [batch, N, maskH, maskW]
        mask_preds = torch.sigmoid(mask_logits)

        # For each batch
        class_losses = []
        score_losses = []
        mask_losses = []
        for batch_idx in range(batch):
            K = len(targets[batch_idx]) # number of targets per batch

            c = class_logits[batch_idx] # [N, C]
            s = score_logits[batch_idx] # [N, 1]
            ml = mask_logits[batch_idx] # [N, H, W]
            mp = mask_preds[batch_idx] # [N, H, W]

            if K == 0:
                # Calculate background loss
                class_loss = self.calculate_bg_class_loss(c)
                class_losses.append(class_loss)
                continue

            label_targets = torch.as_tensor([targets[batch_idx][target_idx]['class_labels'] for target_idx in range(K)], dtype=torch.long, device=device) # [K]
            mask_targets = torch.stack([torch.from_numpy(targets[batch_idx][target_idx]['segmentation']).to(dtype).to(device) for target_idx in range(K)]) # [K, imageH, imageW]

            # Downsample target mask to 1/4 of input size
            mask_targets = F.avg_pool2d(mask_targets, kernel_size=4, stride=4, padding=0) # [N, maskH, maskW]

            score_matrix = self.generate_score_matrix(c, mp, label_targets, mask_targets) # [N, K]
            assigned_inst_idxs, assigned_target_idxs = self.assign_targets_to_instances(score_matrix) # List of length min(N, K)

            # Calculate class loss
            class_loss = self.calculate_class_loss(assigned_inst_idxs, assigned_target_idxs, c, label_targets) # []
            class_losses.append(class_loss)

            # Calculate score loss
            score_loss = self.calculate_score_loss(assigned_inst_idxs, assigned_target_idxs, s, mp, mask_targets) # [min(N, K)]
            score_losses.append(score_loss)

            # Calculate mask loss
            mask_loss = self.calculate_mask_loss(assigned_inst_idxs, assigned_target_idxs, ml, mp, mask_targets) # [min(N, K)]
            mask_losses.append(mask_loss)

        # NOTE: Divide loss by total number of targets across batches to avoid overfitting to images with small number of targets
        total_K = sum([len(t) for t in targets])
        if total_K == 0:
            class_loss = torch.stack(class_losses).mean()
            score_loss = torch.as_tensor(0).to(score_logits)
            mask_loss = torch.as_tensor(0).to(mask_logits)
        else:
            class_loss = torch.stack(class_losses).sum() / total_K
            score_loss = torch.cat(score_losses).sum() / total_K
            mask_loss = torch.cat(mask_losses).sum() / total_K
        return class_loss, score_loss, mask_loss

    def generate_score_matrix(self, class_logits, mask_preds, label_targets, mask_targets, alpha=0.8, eps=1e-6):
        """
        Params:
            class_logits: Tensor[N, C]
            mask_preds: Tensor[N, maskH, maskW]
            label_targets: Tensor[K]
            mask_targets: Tensor[K, maskH, maskW]
            alpha: weighting factor to balance class vs mask
            eps:
        Returns:
            score_matrix: ndarray[N, K]
        """
        N, _, _ = mask_preds.shape
        K, _, _ = mask_targets.shape

        class_preds = torch.sigmoid(class_logits) # [N, C]
        class_scores = class_preds[:, label_targets] # [N, K]

        dice_scores = dice_score_matrix(mask_preds, mask_targets) # [N, K]

        score_matrix = class_scores**(1-alpha) * dice_scores**alpha # [N, K]
        score_matrix += eps

        # Convert to numpy arrays
        score_matrix = score_matrix.detach().cpu().numpy()
        return score_matrix

    def assign_targets_to_instances(self, score_matrix):
        """
        Params:
            score_matrix: ndarray[N, K]
        Returns:
            inst_idxs: List of length min(N, K)
            target_idxs: List of length min(N, K)
        """
        # NOTE: scipy implementation of Hungarian algorithm
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        inst_idxs, target_idxs = scipy.optimize.linear_sum_assignment(score_matrix, maximize=True)
        inst_idxs = inst_idxs.tolist()
        target_idxs = target_idxs.tolist()
        return inst_idxs, target_idxs

    def calculate_bg_class_loss(self, class_logits):
        """
        Params:
            class_logits: Tensor[N, C]
        Returns:
            class_loss: Tensor[]
        """
        N, C = class_logits.shape
        dtype = class_logits.dtype
        device = class_logits.device

        stack_class_targets = torch.zeros(N, C, dtype=dtype, device=device)
        class_loss = sigmoid_focal_loss(class_logits, stack_class_targets, reduction="sum")
        return class_loss

    def calculate_class_loss(self, inst_idxs, target_idxs, class_logits, label_targets):
        """
        Params:
            inst_idxs: List of length min(N, K)
            target_idxs: List of length min(N, K)
            class_logits: Tensor[N, C]
            label_targets: Tensor[K]
        Returns:
            class_loss: Tensor[]
        """
        class_targets = torch.zeros_like(class_logits) # [N, C]
        class_targets[inst_idxs, label_targets[target_idxs]] = 1

        class_loss = sigmoid_focal_loss(class_logits, class_targets, reduction="sum")
        return class_loss

    def calculate_score_loss(self, inst_idxs, target_idxs, score_logits, mask_preds, mask_targets, eps=1e-6):
        """
        Params:
            inst_idxs: List of length min(N, K)
            target_idxs: List of length min(N, K)
            score_logits: Tensor[N, 1]
            mask_preds: Tensor[N, maskH, maskW]
            mask_targets: Tensor[K, maskH, maskW]
            eps: float
        Returns:
            class_loss: Tensor[min(N, K)]
        """
        N, maskH, maskW = mask_preds.shape
        dtype = mask_preds.dtype
        device = mask_preds.device

        stack_score_logits = score_logits[inst_idxs,:].view(-1) # [min(N, K)]
        stack_mask_preds = mask_preds[inst_idxs,:,:] # [min(N, K), maskH, maskW]
        stack_mask_targets = mask_targets[target_idxs,:,:] # [min(N, K), maskH, maskW]

        # Mask-IoU
        stack_mask_preds = (stack_mask_preds > 0.3).float()
        stack_mask_targets = (stack_mask_targets > 0.3).float()
        intersection = (stack_mask_preds * stack_mask_targets).sum(dim=(1,2))
        union = stack_mask_preds.sum(dim=(1,2)) + stack_mask_targets.sum(dim=(1,2)) - intersection
        score_target = intersection / (union + eps)

        score_loss = F.binary_cross_entropy_with_logits(stack_score_logits, score_target.detach(), reduction='none')
        return score_loss

    def calculate_mask_loss(self, inst_idxs, target_idxs, mask_logits, mask_preds, mask_targets):
        """
        Params:
            inst_idxs: List of length min(N, K)
            target_idxs: List of length min(N, K)
            mask_logits: Tensor[N, maskH, maskW]
            mask_preds: Tensor[N, maskH, maskW]
            mask_targets: Tensor[K, maskH, maskW]
        Returns:
            mask_loss: Tensor[min(N, K)]
        """
        stack_mask_logits = mask_logits[inst_idxs,:,:] # [min(N, K), maskH, maskW]
        stack_mask_preds = mask_preds[inst_idxs,:,:] # [min(N, K), maskH, maskW]
        stack_mask_targets = mask_targets[target_idxs,:,:] # [min(N, K), maskH, maskW]
        dice_loss = dice_loss_vector(stack_mask_preds, stack_mask_targets) # [min(N, K)]
        bce_loss = F.binary_cross_entropy_with_logits(stack_mask_logits, stack_mask_targets, reduction='none') # [min(N, K), maskH, maskW]
        mask_loss = dice_loss + bce_loss.mean(dim=(1,2))
        return mask_loss
