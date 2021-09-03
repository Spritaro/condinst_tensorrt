import torch

def heatmap_focal_loss(preds, gt_heatmap, alpha, gamma, eps=1e-3):
    """
    Params:
        preds: Tensor[num_classes, height, width]
        gt_heatmap: Tensor[num_classes, height, width]
        alpha:
        gamma: how much you want to reduce penalty around the ground truth locations
        eps: add small number to prevent inf error
    Returns:
        loss: Tensor[]
    """
    # See CornerNet paper for detail https://arxiv.org/abs/1808.01244
    loss = -torch.where(
        gt_heatmap == 1,
        (1 - preds)**alpha * torch.log(preds + eps), # Loss for positive locations
        (1 - gt_heatmap) ** gamma * (preds)**alpha * torch.log(1 - preds - eps) # loss for negative locations
    ).sum()
    return loss

def dice_loss(inputs, targets, smooth=1.0):
    """
    Params:
        inputs: arbitrary size of Tensor
        targets: arbitrary size of Tensor
        smooth: smoothing factor
    Returns:
        loss: Tensor[]
    """
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Squred denominator version of Dice loss
    dice = (2 * (inputs*targets).sum() + smooth) / ((inputs**2).sum() + (targets**2).sum() + smooth)

    return 1 - dice
