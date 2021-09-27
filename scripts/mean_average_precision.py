import torch
import torch.nn.functional as F

def calc_mask_ious(labels, scores, masks, targets, score_threshold, mask_threshold):
    """
    Params:
        labels: Tensor[num_batch, topk]
        scores: Tensor[num_batch, topk]
        masks: Tensor[num_batch, topk, mask_height, mask_width]
        targets: List[List[Dict{'class_labels': int, 'segmentations': Tensor[image_height, image_width]}]]
        score_threshold: Float
        mask_threshold: Float
    Returns:
        matched_gt_labels: List[Tensor[]]
        matched_scores: List[Tensor[]]
        matched_ious: List[Tensor[]]
    """
    num_batch, topk = scores.shape
    _, _, mask_height, mask_width = masks.shape
    dtype = scores.dtype
    device = scores.device

    masks = (masks > mask_threshold).to(dtype=masks.dtype)

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

            # Assing detected object to GT object
            max_iou = 0.0
            max_iou_gt_i = None
            for gt_i in range(num_objects):

                if gt_i in matched_gt_i:
                    continue

                if labels[batch_i, pred_i] != gt_labels[gt_i]:
                    continue

                inter = torch.sum(masks[batch_i, pred_i, ...] * gt_masks_size_mask[gt_i, ...])
                union = torch.sum(masks[batch_i, pred_i, ...]) + torch.sum(gt_masks_size_mask[gt_i, ...]) - inter
                eps = 1e-6
                iou = inter / (union + eps)

                if iou > max_iou:
                    max_iou = iou
                    max_iou_gt_i = gt_i

                # print("batch_i {} pred_i {} gt_i {} score {} iou {}".format(batch_i, pred_i, gt_i, scores[batch_i, pred_i], iou))

            if max_iou_gt_i is not None:
                matched_gt_i.append(max_iou_gt_i)
                matched_gt_labels.append(gt_labels[max_iou_gt_i])
                matched_scores.append(scores[batch_i, pred_i])
                matched_ious.append(max_iou)

        # print("batch {} GT {} detected {}".format(batch_i, num_objects, len(matched_gt_i)))

    return matched_gt_labels, matched_scores, matched_ious

class MeanAveragePrecision(object):

    def __init__(self, num_classes, score_threshold, mask_threshold):
        self.reset(num_classes)
        self.iou_thresholds = list(range(50, 95, 5)) # AP@[.50:.05:.95] 10 IoU level
        self.recall_thresholds = list(range(0, 101, 1)) # 101 point interpolation
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        return

    def reset(self, num_classes):
        self.ious = [[] for i in range(num_classes)]
        self.num_gts = [0 for i in range(num_classes)]
        return

    def update(self, labels, scores, masks, targets):
        """
        Params:
            labels: Tensor[num_batch, topk]
            scores: Tensor[num_batch, topk]
            masks: Tensor[num_batch, topk, mask_height, mask_width]
            targets: List[List[Dict{'class_labels': int, 'segmentations': Tensor[image_height, image_width]}]]
        Returns:
            None
        """
        gt_labels, scores, ious = calc_mask_ious(
            labels, scores, masks, targets, score_threshold=self.score_threshold, mask_threshold=self.mask_threshold)
        for gt_label, score, iou in zip(gt_labels, scores, ious):
            self.ious[gt_label].append({'score': score, 'iou': iou})

        # Count GTs
        for target in targets:
            gt_labels = torch.as_tensor([obj['class_labels'] for obj in target], dtype=torch.int64)
            for gt_label in gt_labels:
                self.num_gts[gt_label] += 1
        return

    def calc_map(self):
        # For each IoU level...
        aps_per_iou_threshold = []
        for iou_threshold in self.iou_thresholds:

            # For each cateogry...
            aps_per_category = []
            for category_i, (ious, num_gt) in enumerate(zip(self.ious, self.num_gts)):
                # Sort IoUs by score
                ious = sorted(ious, key=lambda k: k['score'], reverse=True)

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

                    # print("TP FP ALL {} {} {}".format(tp, fp, num_gt))

                # Calculate interpolated precisions
                interplated_precisions = []
                max_precision = 0
                recall_i = len(self.recall_thresholds) - 1
                for precision, recall in reversed(list(zip(precisions, recalls))):

                    while recall <= self.recall_thresholds[recall_i] / 100.0:
                        # print("recall_i {} max_precision {} recall {} threshold {}".format(recall_i, max_precision, recall, self.recall_thresholds[recall_i]/100.0))
                        interplated_precisions.append(max_precision)
                        if recall_i <= 0:
                            break
                        recall_i -= 1

                    if precision > max_precision:
                        max_precision = precision

                if len(interplated_precisions) < len(self.recall_thresholds):
                    interplated_precisions.append(max_precision)
                # print("interpolated precisions {}".format(interplated_precisions))

                # Calculate average precision
                ap = sum(interplated_precisions) / len(self.recall_thresholds)
                print("category {} IoU=.{} AP {}".format(category_i, iou_threshold, ap))
                aps_per_category.append(ap)

            # Calculate mAP for each category
            map = sum(aps_per_category) / len(aps_per_category)
            print("IoU=.{} mAP {}".format(iou_threshold, map))
            aps_per_iou_threshold.append(map)

        # Calculate mAP
        map = sum(aps_per_iou_threshold) / len(aps_per_iou_threshold)

        print("AP@[.50:.05:.95] {}".format(map))
        return