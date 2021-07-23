import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.focal_loss import sigmoid_focal_loss

from coco_category_id import category_id_to_label

def get_centroid_indices(masks):
    """
    Params:
        masks: Tensor[num_objects, height, width]
    Returns:
        centroids: Tensor[num_objects, (x, y)]
    """
    _, height, width = masks.shape
    device = masks.device

    location_x = torch.arange(0, width, 1, dtype=torch.float32, device=device) # Tensor[width]
    location_y = torch.arange(0, height, 1, dtype=torch.float32, device=device) # Tensor[height]

    total_area = masks.sum(dim=(1,2)) + 1e-9
    centroids_x = torch.sum(masks.sum(dim=1) * location_x[None,:], dim=1) / total_area # Tensor[num_objects]
    centroids_y = torch.sum(masks.sum(dim=2) * location_y[None,:], dim=1) / total_area # Tensor[num_objects]

    centroids = torch.stack((centroids_x, centroids_y), dim=1) # Tensor[num_objects, (x, y)]
    centroids = centroids.to(torch.int64)
    return centroids

def generate_heatmap(gt_labels, gt_masks, num_classes, sigma=1.0):
    """
    Params:
        gt_labels: Tensor[num_batch]
        gt_masks: Tensor[num_objects, height, width]
        num_classes: int
        sigma: float standard deviation for gaussian distribution
    Returns:
        heatmap: Tensor[num_classes, height, width]
        centroids: Tensor[num_objects, (x, y)]
    """
    num_objects, height, width = gt_masks.shape
    device = gt_masks.device

    centroids = get_centroid_indices(gt_masks) # Tensor[num_objects, (x, y)]

    location_x = torch.arange(0, width, 1, dtype=torch.float32, device=device) # Tensor[width]
    location_y = torch.arange(0, height, 1, dtype=torch.float32, device=device) # Tensor[height]
    location_y, location_x = torch.meshgrid(location_y, location_x) # [height, width], [height, width]

    heatmap = torch.zeros(size=(num_classes, height, width), dtype=torch.float32, device=device)

    for i in range(num_objects):
        label = gt_labels[i]
        px = centroids[i][0]
        py = centroids[i][1]
        # sigma2 = gt_masks[i].sum() * 0.1 + 1e-9
        # sigma2 = float(sigma2.to(torch.device('cpu')).detach().numpy())
        single_heatmap = torch.exp(-((location_x-px)**2 + (location_y-py)**2) / (2. * sigma**2))
        # Take element-wise maximum in case of overlapping objects
        heatmap[label,:,:] = torch.maximum(heatmap[label,:,:], single_heatmap)

    return heatmap, centroids

def heatmap_focal_loss(preds, gt_heatmap, alpha, gamma):
    """
    Params:
        preds: Tensor[num_classes, height, width]
        gt_heatmap: Tensor[num_classes, height, width]
        alpha:
        gamma: how much you want to reduce penalty around the ground truth locations
    Returns:
        loss: Tensor[]
    """
    # See CornerNet paper for detail https://arxiv.org/abs/1808.01244
    loss = -torch.where(
        gt_heatmap == 1,
        (1 - preds)**alpha * torch.log(preds), # Loss for positive locations
        (1 - gt_heatmap) ** gamma * (preds)**alpha * torch.log(1 - preds) # loss for negative locations
    ).sum()
    return loss

def get_heatmap_peaks(cls_logits, topk=100, kernel=3):
    """
    Params:
        cls_logits: Tensor[num_batch, num_classes, height, width]
        topk: Int
        kernel: Int
    Returns:
        keep_labels: Tensor[num_batch, topk]
        keep_cls_preds: Tensor[num_batch, topk]
        keep_points: Tensor[num_batch, topk, (x, y)]
    """
    num_batch, num_classes, height, width = cls_logits.shape
    device = cls_logits.device

    # Get peak maps
    heatmap_preds = cls_logits.sigmoid() # Tensor[num_batch, num_classes, height, width]
    pad = (kernel - 1) // 2
    heatmap_max = F.max_pool2d(heatmap_preds, (kernel, kernel), stride=1, padding=pad) # Tensor[num_batch, num_classes, height, width]
    peak_map = (heatmap_max == heatmap_preds).float()
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

def dice_loss(inputs, targets, smooth=1):
    """
    Params:
        inputs: arbitrary size of Tensor
        targets: arbitrary size of Tensor
        smooth: smoothing factor, default 1
    Returns:
        loss: Tensor[]
    """
    #flatten inputs and targets tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

    return 1 - dice

class CenterNet(nn.Module):
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

        # self.backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=5)
        self.backbone = torchvision.models.resnet50(pretrained=True)

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(num_features=64),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, padding=0)
        )

        self.ctr_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=num_channels, kernel_size=1, padding=0)
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=self.num_filters, kernel_size=1, padding=0)
        )

        # Initialize weight and bias for class head
        nn.init.xavier_uniform_(self.cls_head[-1].weight)
        prior_prob = 0.01
        bias = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head[-1].bias, bias)

    def forward(self, images):
        images = images.to(dtype=torch.float32)

        # features = self.backbone(images)
        # x = list(features.values())[0]

        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x) # 1/4
        x = self.backbone.layer2(x) # 1/8
        x = self.backbone.layer3(x) # 1/16
        x = self.backbone.layer4(x) # 1/32

        x = self.upsample(x) # -> 1/16 -> 1/8

        cls_logits = self.cls_head(x) # [num_batch, num_classes, h, w]
        ctr_logits = self.ctr_head(x) # [num_batch, num_channels, h, w]
        mask_logits = self.mask_head(x) # [num_batch, num_filters, h, w]

        if self.mode == 'training':
            return cls_logits, ctr_logits, mask_logits
        else:
            labels, preds, points = get_heatmap_peaks(cls_logits, topk=self.topk)
            num_batch, num_objects, _ = points.shape
            masks = []
            for i in range(num_batch):
                mask = self.generate_mask(ctr_logits[i], mask_logits[i], points[0])
                masks.append(mask)
            masks = torch.stack(masks, dim=0)
            return labels, preds, masks

    def generate_mask(self, ctr_logits, mask_logits, centroids):
        """
        Params:
            ctr_logits: Tensor[num_channels, height, width]
            mask_logits: Tensor[num_filters, height, width]
            centroids: Tensor[num_objects, (x, y)]
        Returns:
            masks: Tensor[num_objects, height, width]
        """
        _, height, width = ctr_logits.shape
        num_objects, _ = centroids.shape
        device = ctr_logits.device

        # Absolute coordinates
        # NOTE: TensorRT7 does not support float range operation. Use cast instead.
        location_x = torch.arange(0, width, 1, dtype=torch.int32, device=device).float() # Tensor[width]
        location_y = torch.arange(0, height, 1, dtype=torch.int32, device=device).float() # Tensor[height]
        location_y, location_x = torch.meshgrid(location_y, location_x) # Tensor[height, width], Tensor[height, width]
        location_xs = location_x[None,:,:].repeat(num_objects, 1, 1) # Tensor[num_objects, height, width]
        location_ys = location_y[None,:,:].repeat(num_objects, 1, 1) # Tensor[num_objects, height, width]

        # Relative coordinates
        location_xs -= centroids[:,0].view(-1,1,1) # Tensor[num_objects, height, width]
        location_ys -= centroids[:,1].view(-1,1,1) # Tensor[num_objects, height, width]

        # Add relative coordinates to mask features
        mask_logits = mask_logits[None,:,:,:].repeat(num_objects,1,1,1) # Tensor[num_objects, num_filters, height, width]
        mask_logits = torch.cat([mask_logits, location_xs[:,None,:,:], location_ys[:,None,:,:]], dim=1) # Tensor[num_objects, num_filters+2, height, width]

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
            x = mask_logits[no_obj:no_obj+1,:,:,:] # Tensor[1, num_filters+2, height, width]

            x = x.permute(2, 3, 0, 1) # Tensor[height, width, 1, num_filters+2]
            weights1 = weights1.permute(2, 3, 1, 0) # Tensor[1, 1, num_filters+2, num_filters]
            x = torch.matmul(x, weights1) # Tensor[height, width, 1, num_filters]
            x = x + biases1[None, None, None, :]
            x = F.relu(x)

            weights2 = weights2.permute(2, 3, 1, 0) # Tensor[1, 1, num_filters, num_filters]
            x = torch.matmul(x, weights2) # Tensor[height, width, 1, num_filters]
            x = x + biases2[None, None, None, :]
            x = F.relu(x)

            weights3 = weights3.permute(2, 3, 1, 0) # Tensor[1, 1, num_filters, 1]
            x = torch.matmul(x, weights3) # Tensor[height, width, 1, 1]
            x = x + biases3[None, None, None, :]

            x = x.permute(2, 3, 0, 1) # Tensor[1, 1, height, width]
            x = x.view(1, height, width) # Tensor[1, height, width]
            mask = torch.sigmoid(x)

            masks.append(mask)
        masks = torch.cat(masks, dim=0)
        return masks

    def loss(self, cls_logits, ctr_logits, mask_logits, targets):
        """
        Params:
            cls_logits: Tensor[num_batch, num_classes, feature_height, feature_width]
            ctr_logits: Tensor[num_batch, num_channels, feature_height, feature_width]
            mask_logits: Tensor[num_batch, num_filters, feature_height, feature_width]
            targets: List[List[Dict{'category_id': int, 'segmentations': Tensor[image_height, image_width]}]]
        Returns:
            heatmap_loss: Tensor[]
            mask_loss: Tensor[]
        """
        num_batch, num_classes, feature_height, feature_width = cls_logits.shape
        device = cls_logits.device

        # Assign each GT mask to one point in feature map, then calculate loss
        heatmap_losses = []
        mask_losses = []
        for i in range(num_batch):

            # Skip if no object in targets
            if len(targets[i]) == 0:
                heatmap_losses.append(torch.tensor(0, dtype=torch.float32, device=device))
                mask_losses.append(torch.tensor(0, dtype=torch.float32, device=device))
                continue

            # Convert list of dicts to Tensors
            gt_labels = torch.as_tensor([category_id_to_label[obj['category_id']] for obj in targets[i]], dtype=torch.int64, device=device) # Tensor[num_objects]
            gt_masks = torch.stack([torch.as_tensor(obj['segmentation'], dtype=torch.float32, device=device) for obj in targets[i]], dim=0) # Tensor[num_objects, image_height, image_width]

            # Downsample GT masks
            gt_masks_resized = F.interpolate(gt_masks[None,...], size=(feature_height, feature_width)) # Tensor[1, num_objects, feature_height, feature_width]
            gt_masks_resized = gt_masks_resized[0,...] # Tensor[num_objects, feature_height, feature_width]

            # Generate GT heatmap
            gt_heatmap, gt_centroids = generate_heatmap(gt_labels, gt_masks_resized, num_classes) # Tensor[num_objects, feature_height, feature_width], Tensor[num_objects, (x, y)]

            # Generate mask for each object
            masks = self.generate_mask(ctr_logits[i], mask_logits[i], gt_centroids) # Tensor[num_objects, feature_height, feature_width]

            # Calculate loss
            num_objects, _, _ = gt_masks.shape

            heatmap_loss = heatmap_focal_loss(cls_logits[i].sigmoid(), gt_heatmap, alpha=2, gamma=4) / num_objects

            masks = F.interpolate(masks[None,...], scale_factor=4, mode='bilinear') # Tensor[num_objects, image_height/2, image_width/2]
            gt_masks_resized = F.interpolate(gt_masks[None,...], scale_factor=0.5, mode='bilinear') # Tensor[num_objects, image_height/2, image_width/2]
            mask_loss = dice_loss(masks, gt_masks_resized) / num_objects

            heatmap_losses.append(heatmap_loss)
            mask_losses.append(mask_loss)

        heatmap_loss =torch.stack(heatmap_losses, dim=0).mean()
        mask_loss = torch.stack(mask_losses).mean()
        return heatmap_loss, mask_loss

