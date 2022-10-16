import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from sparseinst import SparseInst
from loss import SparseInstLoss
from mean_average_precision import MeanAveragePrecision
from loss_logger import LossLogger

class LitSparseInst(pl.LightningModule):
    def __init__(self, mode, input_height, input_width, input_channels, num_classes, num_instances,
                learning_rate=None, warmup_steps=100, score_threshold=0.3, mask_threshold=0.5,
                class_loss_factor=2.0, score_loss_factor=1.0, mask_loss_factor=2.0):
        super().__init__()
        self.num_instances = num_instances
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.class_loss_factor = class_loss_factor
        self.score_loss_factor = score_loss_factor
        self.mask_loss_factor = mask_loss_factor

        self.sparseinst = SparseInst(mode, input_height, input_width, input_channels, num_classes, num_instances)
        self.sparseinst_loss = SparseInstLoss()

        self.train_loss_logger = LossLogger(class_loss=0, score_loss=0, mask_loss=0, loss=0)
        self.val_loss_logger = LossLogger(val_class_loss=0, val_score_loss=0, val_mask_loss=0, val_loss=0)

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

        class_logits, score_logits, mask_logits, iam = self(images)
        class_loss, score_loss, mask_loss = self.sparseinst_loss(class_logits, score_logits, mask_logits, targets)
        loss = (self.class_loss_factor * class_loss +
                self.score_loss_factor * score_loss +
                self.mask_loss_factor * mask_loss)

        # TensorBoard
        self.train_loss_logger.add_losses(
            class_loss=class_loss, score_loss=score_loss, mask_loss=mask_loss, loss=loss)
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            # Display loss
            self.train_loss_logger.dispaly_losses(self.logger, self.global_step)

            # Display masks
            tensorboard = self.logger.experiment
            mask_preds = torch.sigmoid(mask_logits)
            images_and_masks = self.concat_images_and_masks(images[:,:3,:,:], class_logits, score_logits, mask_preds, iam)
            img_grid = torchvision.utils.make_grid(images_and_masks, nrow=images.shape[0])
            tensorboard.add_image('images', img_grid, global_step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        class_logits, score_logits, mask_logits, iam = self(images)
        class_loss, score_loss, mask_loss = self.sparseinst_loss(class_logits, score_logits, mask_logits, targets)
        loss = (self.class_loss_factor * class_loss +
                self.score_loss_factor * score_loss +
                self.mask_loss_factor * mask_loss)

        # TensorBoard
        self.val_loss_logger.add_losses(
            val_class_loss=class_loss, val_score_loss=score_loss, val_mask_loss=mask_loss, val_loss=loss)
        self.val_loss_logger.dispaly_losses(self.logger, self.global_step)

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
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-6)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        def warmup_lr(epoch):
            if self.trainer.global_step < self.warmup_steps:
                lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.warmup_steps))
            else:
                lr_scale = 1.
            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
        return [optimizer], [scheduler]

    def concat_images_and_masks(self, images, class_logits, score_logits, mask_preds, iam=None):
        """
        Params:
            images: Tensor[batch, 3, imageH, imageW]
            class_logits: Tensor[batch, N, C]
            score_logits: Tensor[batch, N, 1]
            mask_preds: Tensor[batch, N, H, W]
            iam: Tensor[batch, N, H, W]
        Returns:
            images_and_depths: Tensor[batch*2, 3, H, W]
        """
        batch, _, H, W = mask_preds.shape
        device = images.device

        resized_images = F.interpolate(images, size=(H, W))

        def pseudo_color(idx, base=5, channel=0):
            return (float(idx // (base**channel)) % base) / (base - 1)

        inst_idxs_list = [[] for i in range(batch)]

        # Draw mask
        maskmaps = torch.zeros_like(resized_images)
        class_pred = torch.sigmoid(class_logits.max(dim=-1)[0])
        score_pred = torch.sigmoid(score_logits.squeeze(dim=-1))
        score = torch.sqrt(class_pred * score_pred)
        for batch_idx in range(batch):
            for inst_idx in range(self.num_instances):
                if score[batch_idx,inst_idx] > self.score_threshold:
                    maskmaps[batch_idx,0,:,:] = torch.maximum(maskmaps[batch_idx,0,:,:], mask_preds[batch_idx,inst_idx,:,:] * pseudo_color(inst_idx+1, channel=0))
                    maskmaps[batch_idx,1,:,:] = torch.maximum(maskmaps[batch_idx,1,:,:], mask_preds[batch_idx,inst_idx,:,:] * pseudo_color(inst_idx+1, channel=1))
                    maskmaps[batch_idx,2,:,:] = torch.maximum(maskmaps[batch_idx,2,:,:], mask_preds[batch_idx,inst_idx,:,:] * pseudo_color(inst_idx+1, channel=2))
                    inst_idxs_list[batch_idx].append(inst_idx)

        # Draw iam
        max_num_iam = 5
        iammaps = [torch.ones_like(resized_images) for _ in range(min(self.num_instances, max_num_iam))]
        for batch_idx in range(batch):
            inst_idxs = inst_idxs_list[batch_idx][:max_num_iam]
            for i, inst_idx in enumerate(inst_idxs):
                iammap = iammaps[i]
                iammap[batch_idx,1,:,:] = (1-iam[batch_idx,inst_idx,:,:]*20)
                iammap[batch_idx,2,:,:] = (1-iam[batch_idx,inst_idx,:,:]*20)

        # Unnormalize images
        factor = torch.as_tensor([0.229, 0.224, 0.225], device=device)[None,:,None,None].expand(resized_images.shape)
        offset = torch.as_tensor([0.485, 0.456, 0.406], device=device)[None,:,None,None].expand(resized_images.shape)

        images_and_masks = torch.cat([resized_images * factor + offset, maskmaps, *iammaps])
        return images_and_masks