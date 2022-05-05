import torch

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
