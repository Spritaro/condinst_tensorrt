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
        sigma2 = gt_masks[i].sum() * 0.1 + 1e-9
        sigma2 = float(sigma2.to(torch.device('cpu')).detach().numpy())
        single_heatmap = torch.exp(-((location_x-px)**2 + (location_y-py)**2) / (2. * sigma2))
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
    def __init__(self, training, num_classes, topk=100):
        super().__init__()
        self.training = training
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

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, padding=1)
        )

        self.ctr_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=1, padding=1)
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.num_filters, kernel_size=1, padding=1)
        )

        # Initialize weight and bias for class head
        nn.init.xavier_uniform(self.cls_head[-1].weight)
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

        x = self.upsample(x) # -> 1/16 -> 1/8 -> 1/4

        cls_logits = self.cls_head(x) # [num_batch, num_classes, h, w]
        ctr_logits = self.ctr_head(x) # [num_batch, num_channels, h, w]
        mask_logits = self.mask_head(x) # [num_batch, num_filters, h, w]

        if self.training:
            return cls_logits, ctr_logits, mask_logits
        else:
            cls_preds = self.keep_topk(cls_logits, self.topk)
            return cls_preds

    def loss(self, cls_logits, ctr_logits, mask_logits, targets):
        """
        Params:
            cls_logits: Tensor[num_batch, num_classes, feature_height, feature_width]
            ctr_logits: Tensor[num_batch, num_channels, feature_height, feature_width]
            mask_logits: Tensor[num_batch, num_filters, feature_height, feature_width]
            targets: List[List[Dict{'category_id': int, 'segmentations': Tensor[image_height, image_width]}]]
        Returns:
            loss: Tensor[num_batch]
        """
        num_batch, num_classes, feature_height, feature_width = cls_logits.shape
        device = cls_logits.device

        # Assign each GT mask to one point in feature map, then calculate loss
        losses = []
        for i in range(num_batch):

            # Skip if no object in targets
            if len(targets[i]) == 0:
                losses.append(torch.tensor(0, dtype=torch.float32, device=device))
                continue

            # Convert list of dicts to Tensors
            gt_labels = torch.as_tensor([category_id_to_label[obj['category_id']] for obj in targets[i]], dtype=torch.int64, device=device) # Tensor[num_objects]
            gt_masks = torch.stack([torch.as_tensor(obj['segmentation'], dtype=torch.float32, device=device) for obj in targets[i]], dim=0) # Tensor[num_objects, image_height, image_width]

            # Downsample GT masks
            gt_masks = F.interpolate(gt_masks[None,...], size=(feature_height, feature_width)) # Tensor[1, num_objects, feature_height, feature_width]
            gt_masks = gt_masks[0,...] # Tensor[num_objects, feature_height, feature_width]

            # Generate GT heatmap
            gt_heatmap, gt_centroids = generate_heatmap(gt_labels, gt_masks, num_classes) # Tensor[num_objects, feature_height, feature_width], Tensor[num_objects, (x, y)]

            # Generate mask for each object
            mask_losses = []
            num_objects, feature_height, feature_width = gt_masks.shape
            for no_obj in range(num_objects):
                # Centroid point
                px = gt_centroids[no_obj,0]
                py = gt_centroids[no_obj,1]

                # Add relative coordinates to mask features
                location_x = torch.arange(-px, feature_width-px, 1, dtype=torch.float32, device=device) # Tensor[feature_width]
                location_y = torch.arange(-py, feature_height-py, 1, dtype=torch.float32, device=device) # Tensor[feature_height]
                location_y, location_x = torch.meshgrid(location_y, location_x) # [feature_height, feature_width], [feature_height, feature_width]
                x = torch.cat([mask_logits[i,:,:,:], location_x[None,:,:], location_y[None,:,:]], dim=0) # Tensor[num_filters+2, feature_height, feature_width]
                x = x[None,:,:,:]

                # Create instance-aware mask head
                weights1 = ctr_logits[i, :self.conv1_w, py, px].view(self.num_filters,self.num_filters+2,1,1)
                weights2 = ctr_logits[i, self.conv1_w:self.conv2_w, py, px].view(self.num_filters,self.num_filters,1,1)
                weights3 = ctr_logits[i, self.conv2_w:self.conv3_w, py, px].view(1,self.num_filters,1,1)
                biases1 = ctr_logits[i, self.conv3_w:self.conv1_b, py, px]
                biases2 = ctr_logits[i, self.conv1_b:self.conv2_b, py, px]
                biases3 = ctr_logits[i, self.conv2_b:self.conv3_b, py, px]

                # Apply mask head to mask features with relative coordinates
                x = F.conv2d(x, weights1, biases1)
                x = F.relu(x)
                x = F.conv2d(x, weights2, biases2)
                x = F.relu(x)
                x = F.conv2d(x, weights3, biases3)
                x = F.sigmoid(x)

                mask_loss = dice_loss(x, gt_masks[no_obj,:,:])
                mask_losses.append(mask_loss)

            # Calculate loss
            heatmap_loss = heatmap_focal_loss(cls_logits[i].sigmoid(), gt_heatmap, alpha=2, gamma=4) / num_objects
            loss = heatmap_loss + 1.0 * torch.stack(mask_losses).sum() / num_objects
            losses.append(loss)

        return torch.stack(losses, dim=0).mean()

