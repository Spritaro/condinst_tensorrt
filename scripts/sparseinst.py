import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.ops.focal_loss import sigmoid_focal_loss

from loss import dice, dice_loss


def conv1x1_bn(in_channels, out_channels, num_stack=1):
    layers = []
    for i in range(num_stack):
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_channels))
    return nn.Sequential(*layers)

def conv3x3_bn_relu(in_channels, out_channels, num_stack=1):
    layers = []
    for i in range(num_stack):
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.lateral_conv3 = conv1x1_bn(512, num_channels)
        self.lateral_conv4 = conv1x1_bn(1024, num_channels)
        self.lateral_conv5 = conv1x1_bn(2048, num_channels)

        self.conv3 = conv3x3_bn_relu(num_channels, num_channels)
        self.conv4 = conv3x3_bn_relu(num_channels, num_channels)
        self.conv5 = conv3x3_bn_relu(num_channels, num_channels)
        return

    def forward(self, c3, c4, c5):
        # TODO: add Pyarmid Pooling Module

        # FPN
        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.lateral_conv3(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)

        # 3x3 convs
        x5 = self.conv5(p5)
        x4 = self.conv4(p4)
        x3 = self.conv3(p3)

        # Concat
        x5 = F.interpolate(x5, scale_factor=4, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        feature = torch.cat([x3, x4, x5], dim=1)
        return feature


class Decoder(nn.Module):
    def __init__(self, num_classes, num_instances, num_channels):
        super().__init__()

        self.num_classes = num_classes # C
        self.num_instances = num_instances # N
        self.num_channels = num_channels # D

        self.inst_branch = conv3x3_bn_relu(self.num_channels*3+2, self.num_channels, num_stack=4)
        self.mask_branch = conv3x3_bn_relu(self.num_channels*3+2, self.num_channels, num_stack=4)

        self.f_iam = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_instances, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid())

        self.class_head = nn.Sequential(
            nn.Linear(self.num_channels, self.num_classes))
        self.kernel_head = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels))
        # TODO: add score head
        return

    def forward(self, feature):
        """
        Params:
            feature: Tensor[batch, D, N, W]
        Returns:
            class_logits: Tensor[batch, N, C]
            mask_preds: Tensor[batch, N, H, W]
        """
        batch = feature.shape[0]

        inst_feature = self.inst_branch(feature)
        mask_feature = self.mask_branch(feature)

        # Instance activation map
        iam = self.f_iam(inst_feature) # [batch, N, H, W]
        iam = iam.view(batch, self.num_instances, -1) # [batch, N, (H*W)]
        iam = nn.functional.normalize(iam, p=2.0, dim=2)

        # Instance aware feature
        inst_feature = inst_feature.view(batch, self.num_channels, -1) # [batch, D, (H*W)]
        inst_feature = torch.transpose(inst_feature, 1, 2) # [batch, (H*W), D]
        inst_aware_feature = torch.matmul(iam, inst_feature) # [batch, N, D] = [batch, N, (H*W)] * [batch, (H*W), D]

        # Heads
        class_logits = self.class_head(inst_aware_feature) # [batch, N, C]
        kernel_logits = self.kernel_head(inst_aware_feature) # [batch, N, D]

        # Masks
        mask_preds = self.generate_mask(kernel_logits, mask_feature)

        return class_logits, mask_preds

    def generate_mask(self, kernel_logits, mask_feature):
        """
        Params:
            kernel_logits: Tensor[batch, N, D]
            mask_feature: Tensor[batch, D, H, W]
        Returns:
            mask_preds: Tensor[batch, N, H, W]
        """
        batch, N, D = kernel_logits.shape
        _, _, H, W = mask_feature.shape

        m = mask_feature.view(batch, 1, D, 1, -1) # [batch, 1, D, 1, (H*W)]
        m = m.transpose(2, 4) # [batch, 1, (H*W), 1, D]

        w = kernel_logits.view(batch, N, 1, D, 1) # [batch, N, 1, D, 1]

        mask_logits = torch.matmul(m, w) # [batch, N, (H*W), 1, 1] = [batch, 1, (H*W), 1, D] * [batch, N, 1, D, 1]
        mask_logits = mask_logits.view(batch, -1, H, W) # Tensor[batch, N, H, W]
        mask_preds = torch.sigmoid(mask_logits)

        return mask_preds


class SparseInst(nn.Module):
    def __init__(self, mode, input_channels, num_classes, num_instances):
        super().__init__()
        assert mode in ['training', 'inference']
        self.mode = mode
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.num_channels = 256

        self.backbone = torchvision.models.resnet50(pretrained=True)

        self.encoder = Encoder(self.num_channels)
        self.decoder = Decoder(self.num_classes, self.num_instances, self.num_channels)

        def freeze_bn(m):
            if isinstance(m, nn.BatchNorm2d):
                assert(hasattr(m, 'track_running_stats'))
                m.track_running_stats = False
        self.backbone.apply(freeze_bn)

        def initialize(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.encoder.apply(initialize)
        self.decoder.apply(initialize)

        # # Initialize last layer of class head
        # # NOTE: see Focal Loss paper for detail https://arxiv.org/abs/1708.02002
        # pi = 0.01
        # bias = -math.log((1 - pi) / pi)
        # nn.init.constant_(self.decoder.class_head[-1].bias, bias)

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
        class_logits, mask_preds = self.decoder(feature)

        if self.mode == 'training':
            return class_logits, mask_preds
        else:
            class_preds = torch.sigmoid(class_logits)
            labels = torch.argmax(class_preds, dim=2)
            return labels.int(), class_preds.float(), mask_preds.float()

    def add_coordinate(self, feature):
        batch, D, H, W = feature.shape
        dtype = feature.dtype
        device = feature.device

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

    def loss(self, class_logits, mask_preds, targets):
        # TODO: add score loss
        """
        Params:
            class_logits: Tensor[batch, N, C]
            mask_preds: Tensor[batch, N, H, W]
            targets: List[List[Dict{'class_labels': int, 'segmentation': ndarray[imageH, imageW]}]]
        Returns:
            class_loss: Tensor[]
            mask_loss: Tensor[]
        """
        batch, N, C = class_logits.shape
        _, _, H, W = mask_preds.shape
        dtype = class_logits.dtype
        device = class_logits.device

        # Upsample masks to 1/4 of input size
        mask_preds = F.interpolate(mask_preds, scale_factor=2, mode='bilinear', align_corners=False) # [batch, N, maskH, maskW]

        # For each batch
        class_losses = []
        mask_losses = []
        for batch_idx in range(batch):
            K = len(targets[batch_idx]) # num_targets

            c = class_logits[batch_idx] # [N, C]
            m = mask_preds[batch_idx] # [N, H, W]

            if K == 0:
                # Calculate background loss
                class_loss = self.calculate_bg_class_loss(c)
                class_losses.append(class_loss)
                mask_loss = torch.as_tensor(0.0, dtype=dtype, device=device)
                mask_losses.append(mask_loss)
                continue

            label_targets = torch.as_tensor([targets[batch_idx][target_idx]['class_labels'] for target_idx in range(K)], dtype=torch.long, device=device) # [K]
            mask_targets = torch.stack([torch.from_numpy(targets[batch_idx][target_idx]['segmentation']).to(dtype).to(device) for target_idx in range(K)]) # [K, imageH, imageW]

            # Downsample target mask to 1/4 of input size
            mask_targets = F.avg_pool2d(mask_targets, kernel_size=4, stride=4, padding=0) # [batch, N, maskH, maskW]

            score_matrix = self.generate_score_matrix(c, m, label_targets, mask_targets) # [N, K]
            assigned_inst_idxs, assigned_target_idxs = self.assign_targets_to_instances(score_matrix) # List of length min(N, K)

            # Calculate class loss
            class_loss = self.calculate_class_loss(assigned_inst_idxs, assigned_target_idxs, c, label_targets)
            class_losses.append(class_loss)

            # Calculate mask loss
            mask_loss = self.calculate_mask_loss(assigned_inst_idxs, assigned_target_idxs, m, mask_targets)
            mask_losses.append(mask_loss)

        class_loss = torch.stack(class_losses).mean()
        mask_loss = torch.stack(mask_losses).mean()
        return class_loss, mask_loss

    def generate_score_matrix(self, class_logits, mask_preds, label_targets, mask_targets, alpha=0.8, eps=1e-3):
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
        device = class_logits.device

        class_preds = torch.sigmoid(class_logits) # [N, C]

        list_class_preds = []
        list_mask_preds = []
        list_mask_targets = []
        for inst_idx in range(N):
            for target_idx in range(K):
                list_class_preds.append(class_preds[inst_idx, label_targets[target_idx]])
                list_mask_preds.append(mask_preds[inst_idx,:,:])
                list_mask_targets.append(mask_targets[target_idx,:,:])
        stack_class_preds = torch.stack(list_class_preds) # [(N*K)]
        stack_mask_preds = torch.stack(list_mask_preds) # [(N*K), maskH, maskW]
        stack_mask_targets = torch.stack(list_mask_targets) # [(N*K), maskH, maskW]

        score = stack_class_preds**(1-alpha) * dice(stack_mask_preds, stack_mask_targets)**alpha # [(N*K)]
        score_matrix = score.view(N, K)
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
        N, K = score_matrix.shape
        dtype = score_matrix.dtype

        min_score = 0.0
        max_score = np.max(score_matrix)

        inst_idxs = []
        target_idxs = []
        while max_score > min_score:
            arg_max = np.argmax(score_matrix.flatten())
            inst_idx = int(arg_max / K)
            target_idx = int(arg_max % K)

            score_matrix[inst_idx, :] = min_score
            score_matrix[:, target_idx] = min_score

            max_score = np.max(score_matrix)

            inst_idxs.append(inst_idx)
            target_idxs.append(target_idx)
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
        class_loss = sigmoid_focal_loss(class_logits, stack_class_targets, alpha=0.2, reduction="mean")
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
        N, C = class_logits.shape
        dtype = class_logits.dtype
        device = class_logits.device

        zero_hot_label = torch.zeros(C, device=device)

        list_class_targets = []
        for inst_idx in range(N):
            if inst_idx in inst_idxs:
                # Foreground instance
                pair_idx = inst_idxs.index(inst_idx)
                target_idx = target_idxs[pair_idx]
                one_hot_label = nn.functional.one_hot(label_targets[target_idx], num_classes=C).to(dtype) # [1, C]
                one_hot_label = one_hot_label.view(C) # [C]
                list_class_targets.append(one_hot_label)
            else:
                # Background instance
                list_class_targets.append(zero_hot_label)
        stack_class_targets = torch.stack(list_class_targets) # [N, C]
        class_loss = sigmoid_focal_loss(class_logits, stack_class_targets, alpha=0.2, reduction="mean")
        return class_loss

    def calculate_mask_loss(self, inst_idxs, target_idxs, mask_preds, mask_targets):
        """
        Params:
            inst_idxs: List of length min(N, K)
            target_idxs: List of length min(N, K)
            mask_preds: Tensor[N, maskH, maskW]
            mask_targets: Tensor[K, maskH, maskW]
        Returns:
            class_loss: Tensor[]
        """
        stack_mask_preds = mask_preds[inst_idxs,:,:] # [min(N, K), maskH, maskW]
        stack_mask_targets = mask_targets[target_idxs,:,:] # [min(N, K), maskH, maskW]
        mask_loss = dice_loss(stack_mask_preds, stack_mask_targets) # [min(N, K)]
        return mask_loss.mean()