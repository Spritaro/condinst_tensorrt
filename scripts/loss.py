import scipy

import torch
import torch.nn.functional as F
from torch import nn

from torchvision.ops.focal_loss import sigmoid_focal_loss

def dice_score_matrix(inputs, targets, eps=1e-6):
    """
    Params:
        inputs: Tensor[N, *]
        targets: Tensor[K, *]
        eps:
    Returns:
        dice: Tensor[N, K]
    """
    N = inputs.shape[0]
    K = targets.shape[0]
    inputs = inputs.view(N, -1)
    targets = targets.view(K, -1)

    matrix = torch.matmul(inputs, targets.t()) # [N, K]
    inputs2 = (inputs*inputs).sum(dim=1) # [N]
    targets2 = (targets*targets).sum(dim=1) # [K]

    dice = (2 * matrix) / (inputs2.view(N, 1) + targets2.view(1, K) + eps) # [N, K]
    return dice

def dice_loss_vector(inputs, targets, eps=1e-6):
    """
    Params:
        inputs: Tensor[N, *]
        targets: Tensor[N, *]
        smooth:
    Returns:
        dice_loss: Tensor[N]
    """
    N = inputs.shape[0]
    inputs = inputs.view(N, -1)
    targets = targets.view(N, -1)

    dice = (2 * (inputs*targets).sum(dim=1)) / ((inputs*inputs).sum(dim=1) + (targets*targets).sum(dim=1) + eps)
    return 1 - dice


class SparseInstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, class_logits, score_logits, mask_logits, targets):
        """
        Params:
            class_logits: Tensor[batch, N, C]
            score_logits: Tensor[batch, N, 1]
            mask_logits: Tensor[batch, N, H, W]
            targets: List[List[Dict{'class_labels': int, 'segmentation': ndarray[imageH, imageW]}]]
        Returns:
            class_loss: Tensor[]
            score_loss: Tensor[]
            mask_loss: Tensor[]
        """
        batch, N, C = class_logits.shape
        _, _, H, W = mask_logits.shape
        dtype = class_logits.dtype
        device = class_logits.device

        # Upsample masks to 1/4 of input size
        mask_logits = F.interpolate(mask_logits, scale_factor=2, mode='bilinear', align_corners=False) # [batch, N, maskH, maskW]
        mask_preds = torch.sigmoid(mask_logits)

        # For each batch
        class_losses = []
        score_losses = []
        mask_losses = []
        for batch_idx in range(batch):
            K = len(targets[batch_idx]) # number of targets per batch

            c = class_logits[batch_idx] # [N, C]
            s = score_logits[batch_idx] # [N, 1]
            ml = mask_logits[batch_idx] # [N, H, W]
            mp = mask_preds[batch_idx] # [N, H, W]

            if K == 0:
                # Calculate background loss
                class_loss = self.calculate_bg_class_loss(c)
                class_losses.append(class_loss)
                continue

            label_targets = torch.as_tensor([targets[batch_idx][target_idx]['class_labels'] for target_idx in range(K)], dtype=torch.long, device=device) # [K]
            mask_targets = torch.stack([torch.from_numpy(targets[batch_idx][target_idx]['segmentation']).to(dtype).to(device) for target_idx in range(K)]) # [K, imageH, imageW]

            # Downsample target mask to 1/4 of input size
            mask_targets = F.avg_pool2d(mask_targets, kernel_size=4, stride=4, padding=0) # [K, maskH, maskW]

            score_matrix = self.generate_score_matrix(c, mp, label_targets, mask_targets) # [N, K]
            assigned_inst_idxs, assigned_target_idxs = self.assign_targets_to_instances(score_matrix) # List of length min(N, K)

            # Calculate class loss
            class_loss = self.calculate_class_loss(assigned_inst_idxs, assigned_target_idxs, c, label_targets) # []
            class_losses.append(class_loss)

            # Calculate score loss
            score_loss = self.calculate_score_loss(assigned_inst_idxs, assigned_target_idxs, s, mp, mask_targets) # [min(N, K)]
            score_losses.append(score_loss)

            # Calculate mask loss
            mask_loss = self.calculate_mask_loss(assigned_inst_idxs, assigned_target_idxs, ml, mp, mask_targets) # [min(N, K)]
            mask_losses.append(mask_loss)

        # NOTE: Divide loss by total number of targets across batches to avoid overfitting to images with small number of targets
        total_K = sum([len(t) for t in targets])
        if total_K == 0:
            class_loss = torch.stack(class_losses).mean()
            score_loss = torch.as_tensor(0).to(score_logits)
            mask_loss = torch.as_tensor(0).to(mask_logits)
        else:
            class_loss = torch.stack(class_losses).sum() / total_K
            score_loss = torch.cat(score_losses).sum() / total_K
            mask_loss = torch.cat(mask_losses).sum() / total_K
        return class_loss, score_loss, mask_loss

    def generate_score_matrix(self, class_logits, mask_preds, label_targets, mask_targets, alpha=0.8, eps=1e-6):
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

        class_preds = torch.sigmoid(class_logits) # [N, C]
        class_scores = class_preds[:, label_targets] # [N, K]

        dice_scores = dice_score_matrix(mask_preds, mask_targets) # [N, K]

        score_matrix = class_scores**(1-alpha) * dice_scores**alpha # [N, K]
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
        # NOTE: scipy implementation of Hungarian algorithm
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        inst_idxs, target_idxs = scipy.optimize.linear_sum_assignment(score_matrix, maximize=True)
        inst_idxs = inst_idxs.tolist()
        target_idxs = target_idxs.tolist()
        return inst_idxs, target_idxs

    def calculate_bg_class_loss(self, class_logits):
        """
        Params:
            class_logits: Tensor[N, C]
        Returns:
            class_loss: Tensor[]
        """
        class_targets = torch.zeros_like(class_logits) # [N, C]
        class_loss = sigmoid_focal_loss(class_logits, class_targets, reduction="sum")
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
        class_targets = torch.zeros_like(class_logits) # [N, C]
        class_targets[inst_idxs, label_targets[target_idxs]] = 1
        class_loss = sigmoid_focal_loss(class_logits, class_targets, reduction="sum")
        return class_loss

    def calculate_score_loss(self, inst_idxs, target_idxs, score_logits, mask_preds, mask_targets, eps=1e-6):
        """
        Params:
            inst_idxs: List of length min(N, K)
            target_idxs: List of length min(N, K)
            score_logits: Tensor[N, 1]
            mask_preds: Tensor[N, maskH, maskW]
            mask_targets: Tensor[K, maskH, maskW]
            eps: float
        Returns:
            class_loss: Tensor[min(N, K)]
        """
        N, maskH, maskW = mask_preds.shape
        dtype = mask_preds.dtype
        device = mask_preds.device

        stack_score_logits = score_logits[inst_idxs,:].view(-1) # [min(N, K)]
        stack_mask_preds = mask_preds[inst_idxs,:,:].detach() # [min(N, K), maskH, maskW]
        stack_mask_targets = mask_targets[target_idxs,:,:] # [min(N, K), maskH, maskW]

        # Mask-IoU
        stack_mask_preds = (stack_mask_preds > 0.3).float()
        stack_mask_targets = (stack_mask_targets > 0.3).float()
        intersection = (stack_mask_preds * stack_mask_targets).sum(dim=(1,2))
        union = stack_mask_preds.sum(dim=(1,2)) + stack_mask_targets.sum(dim=(1,2)) - intersection
        score_target = intersection / (union + eps)

        score_loss = F.binary_cross_entropy_with_logits(stack_score_logits, score_target, reduction='none')
        return score_loss

    def calculate_mask_loss(self, inst_idxs, target_idxs, mask_logits, mask_preds, mask_targets):
        """
        Params:
            inst_idxs: List of length min(N, K)
            target_idxs: List of length min(N, K)
            mask_logits: Tensor[N, maskH, maskW]
            mask_preds: Tensor[N, maskH, maskW]
            mask_targets: Tensor[K, maskH, maskW]
        Returns:
            mask_loss: Tensor[min(N, K)]
        """
        stack_mask_logits = mask_logits[inst_idxs,:,:] # [min(N, K), maskH, maskW]
        stack_mask_preds = mask_preds[inst_idxs,:,:] # [min(N, K), maskH, maskW]
        stack_mask_targets = mask_targets[target_idxs,:,:] # [min(N, K), maskH, maskW]
        dice_loss = dice_loss_vector(stack_mask_preds, stack_mask_targets) # [min(N, K)]
        bce_loss = F.binary_cross_entropy_with_logits(stack_mask_logits, stack_mask_targets, reduction='none') # [min(N, K), maskH, maskW]
        mask_loss = dice_loss + bce_loss.mean(dim=(1,2))
        return mask_loss
