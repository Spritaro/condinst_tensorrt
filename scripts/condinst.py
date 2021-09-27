import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.ops.focal_loss import sigmoid_focal_loss

from loss import heatmap_focal_loss
from loss import dice_loss

def get_centroid_indices(masks):
    """
    Params:
        masks: Tensor[num_objects, height, width]
    Returns:
        centroids: Tensor[num_objects, (x, y)]
    """
    _, height, width = masks.shape
    dtype = masks.dtype
    device = masks.device

    location_x = torch.arange(0, width, 1, dtype=dtype, device=device) # Tensor[width]
    location_y = torch.arange(0, height, 1, dtype=dtype, device=device) # Tensor[height]

    total_area = masks.sum(dim=(1,2)) + 1e-9
    centroids_x = torch.sum(masks.sum(dim=1) * location_x[None,:], dim=1) / total_area # Tensor[num_objects]
    centroids_y = torch.sum(masks.sum(dim=2) * location_y[None,:], dim=1) / total_area # Tensor[num_objects]

    centroids = torch.stack((centroids_x, centroids_y), dim=1) # Tensor[num_objects, (x, y)]
    centroids = centroids.to(torch.int64)
    return centroids

def generate_heatmap(gt_labels, gt_masks, num_classes):
    """
    Params:
        gt_labels: Tensor[num_objects]
        gt_masks: Tensor[num_objects, height, width]
        num_classes:
    Returns:
        heatmap: Tensor[num_classes, height, width]
        centroids: Tensor[num_objects, (x, y)]
    """
    num_objects, height, width = gt_masks.shape
    dtype = gt_masks.dtype
    device = gt_masks.device

    centroids = get_centroid_indices(gt_masks) # Tensor[num_objects, (x, y)]
    radius2 = torch.sum(gt_masks, dim=(1, 2)) / height / width * 10 + 1

    location_x = torch.arange(0, width, 1, dtype=dtype, device=device) # Tensor[width]
    location_y = torch.arange(0, height, 1, dtype=dtype, device=device) # Tensor[height]
    location_y, location_x = torch.meshgrid(location_y, location_x) # [height, width], [height, width]

    heatmap = torch.zeros(size=(num_classes, height, width), dtype=dtype, device=device)

    for i in range(num_objects):
        label = gt_labels[i]
        px = centroids[i][0]
        py = centroids[i][1]
        single_heatmap = torch.exp(-((location_x-px)**2 + (location_y-py)**2) / (2. * radius2[i]))

        # Take element-wise maximum in case of overlapping objects
        heatmap[label,:,:] = torch.maximum(heatmap[label,:,:], single_heatmap)

    return heatmap, centroids

def get_heatmap_peaks(cls_logits, topk=100, kernel=3):
    """
    Params:
        cls_logits: Tensor[num_batch, num_classes, height, width]
        topk: Int
        kernel: Int
    Returns:
        labels: Tensor[num_batch, topk]
        cls_preds: Tensor[num_batch, topk]
        points: Tensor[num_batch, topk, (x, y)]
    """
    num_batch, num_classes, height, width = cls_logits.shape
    device = cls_logits.device

    # Get peak maps
    heatmap_preds = cls_logits.sigmoid() # Tensor[num_batch, num_classes, height, width]
    pad = (kernel - 1) // 2
    heatmap_max = F.max_pool2d(heatmap_preds, (kernel, kernel), stride=1, padding=pad) # Tensor[num_batch, num_classes, height, width]
    peak_map = (heatmap_max == heatmap_preds).to(dtype=heatmap_preds.dtype)
    peak_map = peak_map * heatmap_preds
    peak_map = peak_map.view(num_batch, -1) # Tensor[num_batch, (num_classes*height*width)]

    # Get properties of each peak
    # NOTE: TensorRT7 does not support rounding_mode='floor' for toch.div()
    cls_preds, keep_idx = torch.topk(peak_map, k=topk, dim=1) # [num_batch, topk], [num_batch, topk]
    labels = torch.div(keep_idx, height*width).long() # [num_batch, topk]
    yx_idx = torch.remainder(keep_idx, height*width).long() # [num_batch, topk]
    ys = torch.div(yx_idx, width).long() # [num_batch, topk]
    xs = torch.remainder(yx_idx, width).long() # [num_batch, topk]
    points = torch.stack([xs, ys], dim=2) # Tensor[num_batch, topk, (x,y)]

    return labels, cls_preds, points

class CondInst(nn.Module):
    def __init__(self, mode, num_classes, topk=100):
        super().__init__()
        assert mode in ['training', 'inference']
        self.mode = mode
        self.topk = topk

        self.num_filters = 8
        self.conv1_w = (self.num_filters + 2) * self.num_filters
        self.conv2_w = self.conv1_w + self.num_filters * self.num_filters
        self.conv3_w = self.conv2_w + self.num_filters * 1
        self.conv1_b = self.conv3_w + self.num_filters
        self.conv2_b = self.conv1_b + self.num_filters
        self.conv3_b = self.conv2_b + 1
        num_channels = self.conv3_b

        # self.backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=0)
        # self.backbone = resnet_fpn_backbone('resnet34', pretrained=True, trainable_layers=5)
        self.backbone = torchvision.models.resnet50(pretrained=True)

        self.lateral_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256)
        )
        self.lateral_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256)
        )
        self.lateral_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256)
        )
        self.lateral_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256)
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, padding=0)
        )

        self.ctr_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=num_channels, kernel_size=1, padding=0)
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=self.num_filters, kernel_size=1, padding=0)
        )

        # Initialize
        def initialize(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.lateral_conv2.apply(initialize)
        self.lateral_conv3.apply(initialize)
        self.lateral_conv4.apply(initialize)
        self.lateral_conv5.apply(initialize)
        self.cls_head.apply(initialize)
        self.ctr_head.apply(initialize)
        self.mask_head.apply(initialize)

        # Initialize last layer of class head
        # NOTE: see Focal Loss paper for detail https://arxiv.org/abs/1708.02002
        pi = 0.01
        bias = -math.log((1 - pi) / pi)
        nn.init.constant_(self.cls_head[-1].bias, bias)

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

        # FPN
        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.lateral_conv3(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.lateral_conv2(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)

        x = p3
        cls_logits = self.cls_head(x) # [num_batch, num_classes, feature_height, feature_width]
        ctr_logits = self.ctr_head(x) # [num_batch, num_channels, feature_height, feature_width]

        x = p2
        mask_logits = self.mask_head(x) # [num_batch, num_filters, mask_height, mask_width]

        if self.mode == 'training':
            return cls_logits, ctr_logits, mask_logits
        else:
            labels, scores, points = get_heatmap_peaks(cls_logits, topk=self.topk)
            num_batch, num_objects, _ = points.shape
            masks = []
            for i in range(num_batch):
                mask = self.generate_mask(ctr_logits[i], mask_logits[i], points[i])
                masks.append(mask)
            masks = torch.stack(masks, dim=0)
            return labels, scores, masks

    def generate_mask(self, ctr_logits, mask_logits, centroids):
        """
        Params:
            ctr_logits: Tensor[num_channels, feature_height, feature_width]
            mask_logits: Tensor[num_filters, mask_height, mask_width]
            centroids: Tensor[num_objects, (x, y)]
        Returns:
            masks: Tensor[num_objects, mask_height, mask_width]
        """
        _, feature_height, feature_width = ctr_logits.shape
        _, mask_height, mask_width = mask_logits.shape
        num_objects, _ = centroids.shape
        dtype = ctr_logits.dtype
        device = ctr_logits.device

        # Absolute coordinates
        # NOTE: TensorRT7 does not support float range operation. Use cast instead.
        location_x = torch.arange(0, mask_width, 1, dtype=torch.int32, device=device) # Tensor[mask_width]
        location_y = torch.arange(0, mask_height, 1, dtype=torch.int32, device=device) # Tensor[mask_height]
        location_x = location_x.to(dtype)
        location_y = location_y.to(dtype)
        location_y, location_x = torch.meshgrid(location_y, location_x) # Tensor[mask_height, mask_width], Tensor[mask_height, mask_width]
        location_xs = location_x[None,:,:].repeat(num_objects, 1, 1) # Tensor[num_objects, mask_height, mask_width]
        location_ys = location_y[None,:,:].repeat(num_objects, 1, 1) # Tensor[num_objects, mask_height, mask_width]

        # Relative coordinates
        location_xs -= centroids[:, 0].view(-1, 1, 1) * (mask_width // feature_width) # Tensor[num_objects, mask_height, mask_width]
        location_ys -= centroids[:, 1].view(-1, 1, 1) * (mask_height // feature_height) # Tensor[num_objects, mask_height, mask_width]
        # location_xs /= mask_width
        # location_ys /= mask_height

        # Add relative coordinates to mask features
        mask_logits = mask_logits[None,:,:,:].expand(num_objects, self.num_filters, mask_height, mask_width) # Tensor[num_objects, num_filters, mask_height, mask_width]
        mask_logits = torch.cat([mask_logits, location_xs[:,None,:,:], location_ys[:,None,:,:]], dim=1) # Tensor[num_objects, num_filters+2, mask_height, mask_width]

        # Create instance-aware mask head
        masks = []
        for no_obj in range(num_objects):
            px = centroids[no_obj,0] # Tensor[]
            py = centroids[no_obj,1] # Tensor[]
            weights1 = ctr_logits[:self.conv1_w, py, px].view(self.num_filters, self.num_filters+2, 1, 1)
            weights2 = ctr_logits[self.conv1_w:self.conv2_w, py, px].view(self.num_filters, self.num_filters, 1, 1)
            weights3 = ctr_logits[self.conv2_w:self.conv3_w, py, px].view(1, self.num_filters, 1, 1)
            biases1 = ctr_logits[self.conv3_w:self.conv1_b, py, px]
            biases2 = ctr_logits[self.conv1_b:self.conv2_b, py, px]
            biases3 = ctr_logits[self.conv2_b:self.conv3_b, py, px]

            # Apply mask head to mask features with relative coordinates
            # NOTE: TensorRT7 does not support dynamic filter for conv2d. Use matmul instead.
            x = mask_logits[no_obj:no_obj+1,:,:,:] # Tensor[1, num_filters+2, mask_height, mask_width]

            x = x.permute(2, 3, 0, 1) # Tensor[mask_height, mask_width, 1, num_filters+2]
            weights1 = weights1.permute(2, 3, 1, 0) # Tensor[1, 1, num_filters+2, num_filters]
            x = torch.matmul(x, weights1) # Tensor[mask_height, mask_width, 1, num_filters]
            x = x + biases1[None, None, None, :]
            x = F.relu(x)

            weights2 = weights2.permute(2, 3, 1, 0) # Tensor[1, 1, num_filters, num_filters]
            x = torch.matmul(x, weights2) # Tensor[mask_height, mask_width, 1, num_filters]
            x = x + biases2[None, None, None, :]
            x = F.relu(x)

            weights3 = weights3.permute(2, 3, 1, 0) # Tensor[1, 1, num_filters, 1]
            x = torch.matmul(x, weights3) # Tensor[mask_height, mask_width, 1, 1]
            x = x + biases3[None, None, None, :]

            x = x.permute(2, 3, 0, 1) # Tensor[1, 1, mask_height, mask_width]
            x = x.view(1, mask_height, mask_width) # Tensor[1, mask_height, mask_width]
            mask = torch.sigmoid(x)

            masks.append(mask)
        masks = torch.cat(masks, dim=0)
        return masks

    def loss(self, cls_logits, ctr_logits, mask_logits, targets):
        """
        Params:
            cls_logits: Tensor[num_batch, num_classes, feature_height, feature_width]
            ctr_logits: Tensor[num_batch, num_channels, feature_height, feature_width]
            mask_logits: Tensor[num_batch, num_filters, mask_height, mask_width]
            targets: List[List[Dict{'class_labels': int, 'segmentations': Tensor[image_height, image_width]}]]
        Returns:
            heatmap_loss: Tensor[]
            mask_loss: Tensor[]
        """
        num_batch, num_classes, feature_height, feature_width = cls_logits.shape
        num_batch, num_filters, mask_height, mask_width = mask_logits.shape
        dtype = cls_logits.dtype
        device = cls_logits.device

        # Assign each GT mask to one point in feature map, then calculate loss
        heatmap_losses = []
        mask_losses = []
        for i in range(num_batch):

            # Skip if no object in targets
            if len(targets[i]) == 0:
                heatmap_losses.append(torch.tensor(0, dtype=dtype, device=device))
                mask_losses.append(torch.tensor(0, dtype=dtype, device=device))
                continue

            # Convert list of dicts to Tensors
            gt_labels = torch.as_tensor([obj['class_labels'] for obj in targets[i]], dtype=torch.int64, device=device) # Tensor[num_objects]
            gt_masks = torch.stack([torch.as_tensor(obj['segmentation'], dtype=dtype, device=device) for obj in targets[i]], dim=0) # Tensor[num_objects, image_height, image_width]

            # Downsample GT masks
            gt_masks_size_feature = F.interpolate(gt_masks[None,...], size=(feature_height, feature_width)) # Tensor[1, num_objects, feature_height, feature_width]
            gt_masks_size_feature = gt_masks_size_feature[0,...] # Tensor[num_objects, feature_height, feature_width]

            # Generate GT heatmap
            gt_heatmap, gt_centroids = generate_heatmap(gt_labels, gt_masks_size_feature, num_classes) # Tensor[num_objects, feature_height, feature_width], Tensor[num_objects, (x, y)]

            # Generate mask for each object
            masks = self.generate_mask(ctr_logits[i], mask_logits[i], gt_centroids) # Tensor[num_objects, mask_height, mask_width]

            # Calculate loss
            num_objects, _, _ = gt_masks.shape

            heatmap_loss = heatmap_focal_loss(cls_logits[i].sigmoid(), gt_heatmap, alpha=2, gamma=4) / num_objects

            # gt_masks_size_mask = F.interpolate(gt_masks[None,...], size=(mask_height, mask_width)) # Tensor[1, num_objects, mask_height, mask_width]
            # _, input_height, input_width = gt_masks.shape
            # masks_input_size = F.interpolate(masks[None,...], size=(input_height, input_width), mode='bilinear', align_corners=False) # Tensor[1, num_objects, mask_height, mask_width]
            gt_masks_size_mask = F.adaptive_avg_pool2d(gt_masks[None,...], output_size=(mask_height, mask_width))
            mask_loss = dice_loss(masks, gt_masks_size_mask)

            heatmap_losses.append(heatmap_loss)
            mask_losses.append(mask_loss)

        heatmap_loss =torch.stack(heatmap_losses, dim=0).mean()
        mask_loss = torch.stack(mask_losses).mean()
        return heatmap_loss, mask_loss

