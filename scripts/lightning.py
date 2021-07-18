import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from torch.utils.tensorboard import SummaryWriter

from centernet import CenterNet, generate_heatmap, get_heatmap_peaks
from coco_category_id import category_id_to_label

def concat_images_and_heatmaps(images, cls_logits, targets, label=1):
    """
    Params:
        images: Tensor[num_batch, 3, H, W]
        cls_logits: Tensor[num_batch, num_classes, h, w]
        targets:
    Returns:
        images_and_heatmaps: Tensor[num_batch*2, 3, h, w]
    """
    num_batch, num_classes, feature_height, feature_width = cls_logits.shape
    device = images.device

    resized_images = F.interpolate(images, size=cls_logits.shape[2:]) # [num_batch, 3, H, W] -> [num_batch, 3, h, w]

    # Create colored heatmaps from class predictions
    heatmaps = torch.ones_like(cls_logits[:,label:label+1,:,:], device=device)
    heatmaps = heatmaps.repeat(1, 3, 1, 1) # [num_batch, 1, h, w] -> [num_batch, 3, h, w]

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
        labels, preds, points = get_heatmap_peaks(cls_logits[i:i+1], topk=10)
        for no_obj in range(10):
            heatmaps[i,0,points[0,no_obj,1],points[0,no_obj,0]] = 0

    heatmaps[:,1:2,:,:] -= cls_logits[:,label:label+1,:,:].sigmoid() # subtract green
    heatmaps[:,2:3,:,:] -= cls_logits[:,label:label+1,:,:].sigmoid() # subtract blue

    # Unnormalize images
    factor = torch.as_tensor([0.229, 0.224, 0.225], device=device)[None,:,None,None].expand(resized_images.shape)
    offset = torch.as_tensor([0.485, 0.456, 0.406], device=device)[None,:,None,None].expand(resized_images.shape)

    images_and_heatmaps = torch.cat([resized_images * factor + offset, heatmaps])
    return images_and_heatmaps

class LitCenterNet(pl.LightningModule):
    def __init__(self, training, num_classes, topk=100):
        super().__init__()
        self.centernet = CenterNet(training, num_classes, topk)

        # TensorBoard
        self.training = training
        if self.training:
            self.writer = SummaryWriter()

    def forward(self, images):
        cls_logits = self.centernet(images)
        return cls_logits

    def training_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        cls_logits, ctr_logits, mask_logits = self(images)
        loss = self.centernet.loss(cls_logits, ctr_logits, mask_logits, targets)

        # TensorBoard
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            # Display loss
            self.writer.add_scalar("loss", loss, self.global_step)

            # Display heatmaps
            images_and_heatmaps = concat_images_and_heatmaps(images, cls_logits, targets)
            img_grid = torchvision.utils.make_grid(images_and_heatmaps, nrow=images.shape[0])
            self.writer.add_image('heatmap_images', img_grid, global_step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        cls_logits, ctr_logits, mask_logits = self(images)
        loss = self.onenet.loss(cls_logits, ctr_logits, mask_logits, targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def on_train_end(self):
        # TensorBoard
        if self.training:
            self.writer.close()
