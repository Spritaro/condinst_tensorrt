import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from torch.utils.tensorboard import SummaryWriter

from centernet import CenterNet, generate_heatmap, get_heatmap_peaks
from coco_category_id import category_id_to_label

def concat_images_and_heatmaps(images, cls_logits, ctr_logits, mask_logits, targets, model, label=1):
    """
    Params:
        images: Tensor[num_batch, 3, H, W]
        cls_logits: Tensor[num_batch, num_classes, feature_height, feature_width]
        ctr_logits: Tensor[num_batch, num_channels, feature_height, feature_width]
        mask_logits: Tensor[num_batch, num_filters, mask_height, mask_width]
        targets:
        model:
        label:
    Returns:
        images_and_heatmaps: Tensor[num_batch*2, 3, h, w]
    """
    num_batch, _, feature_height, feature_width = cls_logits.shape
    num_batch, _, mask_height, mask_width = mask_logits.shape
    device = images.device

    resized_images = F.interpolate(images, size=(mask_height, mask_width))

    # Create colored heatmaps from class predictions
    heatmaps = torch.ones(size=(num_batch, 3, feature_height, feature_width), device=device)
    cls_logits_max, _ = torch.max(cls_logits, dim=1)
    heatmaps[:,1,:,:] -= cls_logits_max.sigmoid() # subtract green
    heatmaps[:,2,:,:] -= cls_logits_max.sigmoid() # subtract blue

    maskmaps = torch.zeros_like(resized_images)

    for i in range(num_batch):
        # Skip if no object in targets
        if len(targets[i]) == 0:
            continue

        # # Convert list of dicts to Tensors
        # gt_labels = torch.as_tensor([category_id_to_label[obj['category_id']] for obj in targets[i]], dtype=torch.int64, device=device) # Tensor[num_objects]
        # gt_masks = torch.stack([torch.as_tensor(obj['segmentation'], dtype=torch.float32, device=device) for obj in targets[i]], dim=0) # Tensor[num_objects, image_height, image_width]
        # # Downsample GT masks
        # gt_masks = F.interpolate(gt_masks[None,...], size=(feature_height, feature_width)) # Tensor[1, num_objects, feature_height, feature_width]
        # gt_masks = gt_masks[0,...] # Tensor[num_objects, feature_height, feature_width]
        # gt_heatmaps, _ = generate_heatmap(gt_labels, gt_masks, num_classes)
        # heatmaps[i,0,:,:] -= gt_heatmaps[label,:,:]

        # Draw peak
        topk = 10
        labels, preds, points = get_heatmap_peaks(cls_logits[i:i+1], topk)
        for no_obj in range(topk):
            heatmaps[i,0,points[0,no_obj,1],points[0,no_obj,0]] = 0

        # Draw mask
        masks = model.generate_mask(ctr_logits[i], mask_logits[i], points[0])
        for no_obj in range(topk):
            maskmaps[i,0,:,:] = torch.maximum(maskmaps[i,0,:,:], masks[no_obj] * (float(no_obj+1)%8/7))
            maskmaps[i,1,:,:] = torch.maximum(maskmaps[i,1,:,:], masks[no_obj] * (float(no_obj+1)%4/3))
            maskmaps[i,2,:,:] = torch.maximum(maskmaps[i,2,:,:], masks[no_obj] * (float(no_obj+1)%2/1))

    # Unnormalize images
    factor = torch.as_tensor([0.229, 0.224, 0.225], device=device)[None,:,None,None].expand(resized_images.shape)
    offset = torch.as_tensor([0.485, 0.456, 0.406], device=device)[None,:,None,None].expand(resized_images.shape)

    # Upsample heatmaps to concatenate with masks
    resized_heatmaps = F.interpolate(heatmaps, size=(mask_height, mask_width))

    images_and_heatmaps = torch.cat([resized_images * factor + offset, resized_heatmaps, maskmaps])
    return images_and_heatmaps

class LitCenterNet(pl.LightningModule):
    def __init__(self, mode, num_classes, topk=100, mask_loss_factor=1.0):
        super().__init__()
        self.mask_loss_factor = mask_loss_factor
        self.centernet = CenterNet(mode, num_classes, topk)

        # TensorBoard
        self.mode = mode
        if self.mode == 'training':
            self.writer = SummaryWriter()

    def forward(self, images):
        outputs = self.centernet(images)
        return outputs

    def training_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        cls_logits, ctr_logits, mask_logits = self(images)
        heatmap_loss, mask_loss = self.centernet.loss(cls_logits, ctr_logits, mask_logits, targets)
        loss = heatmap_loss + self.mask_loss_factor * mask_loss

        # TensorBoard
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            # Display loss
            self.writer.add_scalar("heatmap_loss", heatmap_loss, self.global_step)
            self.writer.add_scalar("mask_loss", mask_loss, self.global_step)
            self.writer.add_scalar("loss", loss, self.global_step)

            # Display heatmaps
            images_and_heatmaps = concat_images_and_heatmaps(images, cls_logits, ctr_logits, mask_logits, targets, self.centernet)
            img_grid = torchvision.utils.make_grid(images_and_heatmaps, nrow=images.shape[0])
            self.writer.add_image('heatmap_images', img_grid, global_step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        cls_logits, ctr_logits, mask_logits = self(images)
        heatmap_loss, mask_loss = self.centernet.loss(cls_logits, ctr_logits, mask_logits, targets)
        loss = heatmap_loss + self.mask_loss_factor * mask_loss

        # TensorBoard
        self.writer.add_scalar("val_heatmap_loss", heatmap_loss, self.global_step)
        self.writer.add_scalar("val_mask_loss", mask_loss, self.global_step)
        self.writer.add_scalar("val_loss", loss, self.global_step)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def on_train_end(self):
        # TensorBoard
        if self.mode == 'training':
            self.writer.close()
