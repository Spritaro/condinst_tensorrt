import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from condinst import CondInst, generate_heatmap, get_heatmap_peaks

def calc_mask_ious(labels, scores, masks, targets, score_threshold=0.3):
    """
    Params:
        labels: Tensor[num_batch, topk]
        scores: Tensor[num_batch, topk]
        masks: Tensor[num_batch, topk, mask_height, mask_width]
        targets: List[List[Dict{'class_labels': int, 'segmentations': Tensor[image_height, image_width]}]]
    Returns:
        matched_gt_labels: List[Tensor[]]
        matched_scores: List[Tensor[]]
        matched_ious: List[Tensor[]]
    """
    num_batch, topk = scores.shape
    _, _, mask_height, mask_width = masks.shape
    dtype = scores.dtype
    device = scores.device

    matched_gt_labels = []
    matched_scores = []
    matched_ious = []
    for batch_i in range(num_batch):

        # Skip if no GT object in targets
        if len(targets[batch_i]) == 0:
            continue

        # Convert list of dicts to Tensors
        gt_labels = torch.as_tensor([obj['class_labels'] for obj in targets[batch_i]], dtype=torch.int64, device=device) # Tensor[num_objects]
        gt_masks = torch.stack([torch.as_tensor(obj['segmentation'], dtype=dtype, device=device) for obj in targets[batch_i]], dim=0) # Tensor[num_objects, image_height, image_width]

        # Downsample GT masks
        gt_masks_size_mask = F.interpolate(gt_masks[None,...], size=(mask_height, mask_width)) # Tensor[1, num_objects, mask_height, mask_width]
        gt_masks_size_mask = gt_masks_size_mask[0,...] # Tensor[num_objects, feature_height, feature_width]

        num_objects, _, _ = gt_masks_size_mask.shape
        matched_gt_i = []
        for pred_i in range(topk):

            if scores[batch_i, pred_i] < score_threshold:
                break

            for gt_i in range(num_objects):

                if gt_i in matched_gt_i:
                    continue

                if labels[batch_i, pred_i] != gt_labels[gt_i]:
                    continue

                inter = torch.sum(masks[batch_i, pred_i, ...] * gt_masks_size_mask[gt_i, ...])
                union = torch.sum(masks[batch_i, pred_i, ...]) + torch.sum(gt_masks_size_mask[gt_i, ...]) - inter
                eps = 1e-6
                iou = inter / (union + eps)

                # print("batch_i {} pred_i {} gt_i {} score {} iou {}".format(batch_i, pred_i, gt_i, scores[batch_i, pred_i], iou))

                if iou < 0.5:
                    continue

                matched_gt_i.append(gt_i)
                matched_gt_labels.append(gt_labels[gt_i])
                matched_scores.append(scores[batch_i, pred_i])
                matched_ious.append(iou)
                break

        # print("batch {} GT {} detected {}".format(batch_i, num_objects, len(matched_gt_i)))

    return matched_gt_labels, matched_scores, matched_ious

class LitCondInst(pl.LightningModule):
    def __init__(self, mode, num_classes, learning_rate=1e-4, topk=100, mask_loss_factor=1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.mask_loss_factor = mask_loss_factor
        self.condinst = CondInst(mode, num_classes, topk)

        # TensorBoard
        self.mode = mode

        # mAP calculation
        self.ious = [[] for i in range(num_classes)]
        self.num_gts = [0 for i in range(num_classes)]
        self.iou_thresholds = list(range(50, 95, 5)) # AP@[.50:.05:.95] 10 IoU level
        self.recall_thresholds = list(range(0, 101, 1)) # 101 point interpolation

    def forward(self, images):
        outputs = self.condinst(images)
        return outputs

    def training_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        cls_logits, ctr_logits, mask_logits = self(images)
        heatmap_loss, mask_loss = self.condinst.loss(cls_logits, ctr_logits, mask_logits, targets)
        loss = heatmap_loss + self.mask_loss_factor * mask_loss

        # TensorBoard
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            # Display loss
            tensorboard = self.logger.experiment
            tensorboard.add_scalar("heatmap_loss", heatmap_loss, self.global_step)
            tensorboard.add_scalar("mask_loss", mask_loss, self.global_step)
            tensorboard.add_scalar("loss", loss, self.global_step)
            # Display heatmaps
            images_and_heatmaps = self.concat_images_and_heatmaps(images, cls_logits, ctr_logits, mask_logits)
            img_grid = torchvision.utils.make_grid(images_and_heatmaps, nrow=images.shape[0])
            tensorboard.add_image('images', img_grid, global_step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # images, targets = batch

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        cls_logits, ctr_logits, mask_logits = self(images)
        heatmap_loss, mask_loss = self.condinst.loss(cls_logits, ctr_logits, mask_logits, targets)
        loss = heatmap_loss + self.mask_loss_factor * mask_loss

        # TensorBoard
        tensorboard = self.logger.experiment
        tensorboard.add_scalar("val_heatmap_loss", heatmap_loss, self.global_step)
        tensorboard.add_scalar("val_mask_loss", mask_loss, self.global_step)
        tensorboard.add_scalar("val_loss", loss, self.global_step)

        return loss

    def test_step(self, batch, batch_idx):

        images = [img for img, _ in batch]
        targets = [tgt for _, tgt in batch]
        images = torch.stack(images, dim=0)

        labels, scores, masks = self(images)
        gt_labels, scores, ious = calc_mask_ious(labels, scores, masks, targets)
        for gt_label, score, iou in zip(gt_labels, scores, ious):
            self.ious[gt_label].append({'score': score, 'iou': iou})
            self.num_gts[gt_label] += 1

    def test_epoch_end(self, outputs):

        # For each cateogry...
        aps_per_category = []
        for category_i, (ious, num_gt) in enumerate(zip(self.ious, self.num_gts)):
            # Sort IoUs by score
            ious = sorted(ious, key=lambda k: k['score'], reverse=True)

            # For each IoU level...
            aps_per_iou_threshold = []
            for iou_threshold in self.iou_thresholds:
                # Calculate precisions and recalls
                tp = 0
                fp = 0
                precisions = []
                recalls = []
                for iou in ious:
                    if iou['iou'] > iou_threshold / 100.0:
                        tp += 1
                    else:
                        fp += 1
                    precision = tp / (tp + fp)
                    recall = tp / num_gt
                    precisions.append(precision)
                    recalls.append(recall)

                    # print("TP FP {} {}".format(tp, fp))

                # Calculate interpolated precisions
                interplated_precisions = []
                max_precision = 0
                recall_i = len(self.recall_thresholds) - 1
                for precision, recall in reversed(list(zip(precisions, recalls))):

                    while recall <= self.recall_thresholds[recall_i] / 100.0:
                        # print("recall_i {} max_precision {} recall {} threshold {}".format(recall_i, max_precision, recall, self.recall_thresholds[recall_i]/100.0))
                        interplated_precisions.append(max_precision)
                        recall_i -= 1
                        if recall_i < 0:
                            break

                    if precision > max_precision:
                        max_precision = precision

                if len(interplated_precisions) < len(self.recall_thresholds):
                    interplated_precisions.append(max_precision)
                # print("interpolated precisions {}".format(interplated_precisions))

                # Calculate average precision
                ap = sum(interplated_precisions) / len(self.recall_thresholds)
                print("category {} AP IoU=.{} {}".format(category_i, iou_threshold, ap))
                aps_per_iou_threshold.append(ap)

            # Calculate mAP for each category
            if len(aps_per_iou_threshold) > 0:
                map = sum(aps_per_iou_threshold) / len(aps_per_iou_threshold)
            else:
                map = 0.0
            aps_per_category.append(map)

        # Calculate mAP for multiple categories
        map = sum(aps_per_category) / len(aps_per_category)

        print("AP@[.50:.05:.95] {}".format(map))

        return map

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-3)
        return optimizer

    def concat_images_and_heatmaps(self, images, cls_logits, ctr_logits, mask_logits):
        """
        Params:
            images: Tensor[num_batch, 3, H, W]
            cls_logits: Tensor[num_batch, num_classes, feature_height, feature_width]
            ctr_logits: Tensor[num_batch, num_channels, feature_height, feature_width]
            mask_logits: Tensor[num_batch, num_filters, mask_height, mask_width]
        Returns:
            images_and_heatmaps: Tensor[num_batch*3, 3, h, w]
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
            # Draw peak
            topk = 10
            labels, scores, points = get_heatmap_peaks(cls_logits[i:i+1], topk=topk)
            num_objects, = scores[scores > 0.1].shape
            for no_obj in range(num_objects):
                heatmaps[i,0,points[0,no_obj,1],points[0,no_obj,0]] = 0

            # Draw mask
            masks = self.condinst.generate_mask(ctr_logits[i], mask_logits[i], points[0])
            for no_obj in range(num_objects):
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