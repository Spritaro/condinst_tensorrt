import torch

def dice(inputs, targets, smooth=1e-3):
    """
    Params:
        inputs: Tensor[N, *]
        targets: Tensor[N, *]
        smooth: smoothing factor
    Returns:
        loss: Tensor[N]
    """
    N = inputs.shape[0]
    inputs = inputs.view(N, -1)
    targets = targets.view(N, -1)

    dice_ = (2 * (inputs*targets).sum(dim=1) + smooth) / ((inputs**2).sum(dim=1) + (targets**2).sum(dim=1) + smooth)
    return dice_

def dice_loss(inputs, targets, smooth=1.0):
    """
    Params:
        inputs: Tensor[N, *]
        targets: Tensor[N, *]
        smooth: smoothing factor
    Returns:
        loss: Tensor[N]
    """
    return 1 - dice(inputs, targets, smooth)
