import torch

def dice(inputs, targets, eps=1e-3):
    """
    Params:
        inputs: Tensor[N, *]
        targets: Tensor[N, *]
        eps:
    Returns:
        loss: Tensor[N]
    """
    N = inputs.shape[0]
    inputs = inputs.view(N, -1)
    targets = targets.view(N, -1)

    dice_ = (2 * (inputs*targets).sum(dim=1)) / ((inputs**2).sum(dim=1) + (targets**2).sum(dim=1) + eps)
    return dice_

def dice_loss(inputs, targets, eps=1e-3):
    """
    Params:
        inputs: Tensor[N, *]
        targets: Tensor[N, *]
        smooth:
    Returns:
        loss: Tensor[N]
    """
    return 1 - dice(inputs, targets, eps)
