import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from sparseinst import SparseInst
from mean_average_precision import MeanAveragePrecision

class LitSparseInst(pl.LightningModule):
    def __init__(self, mode, input_channels, num_classes, num_instances,
                learning_rate, score_threshold=0.3, mask_threshold=0.5,
                class_loss_factor=2.0, score_loss_factor=0.0, mask_loss_factor=2.0):
        super().__init__()
        self.num_instances = num_instances
        self.learning_rate = learning_rate
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.class_loss_factor = class_loss_factor
        self.score_loss_factor = score_loss_factor
        self.mask_loss_factor = mask_loss_factor

        self.sparseinst = SparseInst(mode, input_channels, num_classes, num_instances)

        # mAP calculation
        self.map = MeanAveragePrecision(
            num_classes=num_classes, score_threshold=score_threshold, mask_threshold=mask_threshold)

    def forward(self, images):
        outputs = self.sparseinst(images)
        return outputs

    def training_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        class_logits, score_logits, mask_logits = self(images)
        class_loss, score_loss, mask_loss = self.sparseinst.loss(class_logits, score_logits, mask_logits, targets)
        loss = (self.class_loss_factor * class_loss +
                self.score_loss_factor * score_loss +
                self.mask_loss_factor * mask_loss)

        # TensorBoard
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            # Display loss
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("class_loss", class_loss, self.global_step)
            tensorboard.add_scalar("score_loss", score_loss, self.global_step)
            tensorboard.add_scalar("mask_loss", mask_loss, self.global_step)
            tensorboard.add_scalar("loss", loss, self.global_step)
            # Display masks
            mask_preds = torch.sigmoid(mask_logits)
            images_and_masks = self.concat_images_and_masks(images[:,:3,:,:], class_logits, mask_preds)
            img_grid = torchvision.utils.make_grid(images_and_masks, nrow=images.shape[0])
            tensorboard.add_image('images', img_grid, global_step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        class_logits, score_logits, mask_logits = self(images)
        class_loss, score_loss, mask_loss = self.sparseinst.loss(class_logits, score_logits, mask_logits, targets)
        loss = (self.class_loss_factor * class_loss +
                self.score_loss_factor * score_loss +
                self.mask_loss_factor * mask_loss)

        # TensorBoard
        tensorboard = self.logger.experiment
        tensorboard.add_scalar("val_class_loss", class_loss, self.global_step)
        tensorboard.add_scalar("val_score_loss", score_loss, self.global_step)
        tensorboard.add_scalar("val_mask_loss", mask_loss, self.global_step)
        tensorboard.add_scalar("val_loss", loss, self.global_step)

        # ModelCheckpoint metrics
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        labels, scores, masks = self(images)
        self.map.update(labels, scores, masks, targets)

    def test_epoch_end(self, outputs):

        self.map.calc_map()

        return map

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001, nesterov=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4, eps=1e-3)
        return optimizer

    def concat_images_and_masks(self, images, class_logits, mask_preds):
        """
        Params:
            images: Tensor[batch, 3, imageH, imageW]
            class_logits: Tensor[batch, N, C]
            mask_preds: Tensor[batch, N, H, W]
        Returns:
            images_and_depths: Tensor[batch*2, 3, H, W]
        """
        batch, _, H, W = mask_preds.shape
        device = images.device

        resized_images = F.interpolate(images, size=(H, W))

        maskmaps = torch.zeros_like(resized_images)

        # Draw mask
        for batch_idx in range(batch):
            for inst_idx in range(self.num_instances):
                if torch.sigmoid(torch.max(class_logits[batch_idx,inst_idx,:])) > self.score_threshold:
                    maskmaps[batch_idx,0,:,:] = torch.maximum(maskmaps[batch_idx,0,:,:], mask_preds[batch_idx,inst_idx,:,:] * (float(inst_idx+1)%8/7))
                    maskmaps[batch_idx,1,:,:] = torch.maximum(maskmaps[batch_idx,1,:,:], mask_preds[batch_idx,inst_idx,:,:] * (float(inst_idx+1)%4/3))
                    maskmaps[batch_idx,2,:,:] = torch.maximum(maskmaps[batch_idx,2,:,:], mask_preds[batch_idx,inst_idx,:,:] * (float(inst_idx+1)%2/1))

        # Unnormalize images
        factor = torch.as_tensor([0.229, 0.224, 0.225], device=device)[None,:,None,None].expand(resized_images.shape)
        offset = torch.as_tensor([0.485, 0.456, 0.406], device=device)[None,:,None,None].expand(resized_images.shape)

        images_and_masks = torch.cat([resized_images * factor + offset, maskmaps])
        return images_and_masks