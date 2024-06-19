"""IMPORT PACKAGES"""
import torch
from torch import nn
import torch.nn.functional as F

"""""" """""" """""" """""" """""" """""" """""" """"""
"""" DEFINE HELPER FUNCTIONS FOR LOSS FUNCTION"""
"""""" """""" """""" """""" """""" """""" """""" """"""


def construct_loss_function(opt):
    # Define possible choices for classification loss
    if opt.cls_criterion == "BCE":
        cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([opt.cls_criterion_weight], dtype=torch.float32))
    elif opt.cls_criterion == "CE":
        cls_criterion = nn.CrossEntropyLoss()
    elif opt.cls_criterion == "Focal":
        cls_criterion = FocalLoss_Cls(alpha=opt.focal_alpha_cls, gamma=opt.focal_gamma_cls, reduction='mean')
    else:
        raise Exception("Unexpected Classification Loss {}".format(opt.cls_criterion))

    # Define possible choices for segmentation loss (Single Mask)
    if opt.seg_criterion == "Dice":
        seg_criterion = BinaryDiceLoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == "DiceBCE":
        seg_criterion = DiceBCELoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == "DiceBCE_Weak":
        seg_criterion = DiceBCE_WeakSup_Loss(smooth=1e-6, p=1)
    elif opt.seg_criterion == "IoU":
        seg_criterion = IoU_Loss(smooth=1e-6)
    elif opt.seg_criterion == "Focal":
        seg_criterion = FocalLoss(smooth=1e-6, alpha=opt.focal_alpha_seg, gamma=opt.focal_gamma_seg, reduction='mean')
    elif opt.seg_criterion == "DiceFocal":
        seg_criterion = DiceFocalLoss(alpha=opt.focal_alpha_seg, gamma=opt.focal_gamma_seg, smooth=1e-6, p=1)
    elif opt.seg_criterion == 'MSE':
        seg_criterion = MSELoss(smooth=1e-6)
    elif opt.seg_criterion == 'BCE':
        seg_criterion = BCELoss(smooth=1e-6)

    # Define possible choices for segmentation loss (Multi Mask)
    elif opt.seg_criterion == 'JIGS':
        seg_criterion = JIGSLoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == 'JIGSBCE':
        seg_criterion = JIGSBCELoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == 'MultiMaskBCE':
        seg_criterion = MultiMaskBCELoss(smooth=1e-6)
    elif opt.seg_criterion == 'MultiMaskMSE':
        seg_criterion = MultiMaskMSELoss(smooth=1e-6)
    elif opt.seg_criterion == 'MultiMaskDice':
        seg_criterion = MultiMaskDiceLoss(smooth=1e-6, p=1, variant='Regular')
    elif opt.seg_criterion == 'MultiMaskDiceW':
        seg_criterion = MultiMaskDiceLoss(smooth=1e-6, p=1, variant='Weighted')
    elif opt.seg_criterion == 'MultiMaskDiceBCE':
        seg_criterion = MultiMaskDiceBCELoss(smooth=1e-6, p=1, variant='Regular')
    elif opt.seg_criterion == 'MultiMaskDiceBCEW':
        seg_criterion = MultiMaskDiceBCELoss(smooth=1e-6, p=1, variant='Weighted')

    # Define exception for unexpected segmentation loss
    else:
        raise Exception("Unexpected Segmentation loss {}".format(opt.seg_criterion))

    return cls_criterion, seg_criterion


"""""" """""" """""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM CLASSIFICATION LOSS FUNCTIONS """
"""""" """""" """""" """""" """""" """""" """""" """"""


# Custom Focal Loss for Classification
class FocalLoss_Cls(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Focal_Cls class implements the Focal Loss for binary classification tasks.

    Args:
        alpha (float): Weighting factor to balance positive and negative class samples.
                       If alpha is set to -1, no weighting is applied.
                       positive class = alpha, negative class = (1-alpha)
        gamma (float): Focusing parameter controlling the degree of emphasis on hard-to-classify samples.
        reduction (str): Specifies the reduction to apply to the computed loss.
                         Supported values are 'none', 'mean', and 'sum'.

    Attributes:
        alpha (float): Weighting factor for class balancing.
        gamma (float): Focusing parameter for the Focal Loss.
        reduction (str): Reduction mode for the computed loss.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
        __call__(self, preds, targets):
            Computes the Focal Loss based on the provided predictions and targets.

    Example:
        focal_loss = Focal_Cls(alpha=0.25, gamma=2, reduction='mean')
        loss = focal_loss(predictions, targets)
    """

    def __init__(self, alpha, gamma, reduction):
        super(FocalLoss_Cls, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, targets):
        """
        Compute Focal Loss based on the provided predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed Focal Loss based on the specified reduction mode.
        """

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == targets.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Compute Binary Cross Entropy Loss
        bce = F.binary_cross_entropy(preds, targets, reduction='none')

        # Compute Focal Loss
        p_t = preds * targets + (1 - preds) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)

        # Apply the alpha weighting factor
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} "
                f"\n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        return loss


"""""" """""" """""" """""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM SEGMENTATION LOSS FUNCTIONS (SINGLE) """
"""""" """""" """""" """""" """""" """""" """""" """""" """"""


# Custom BCE Loss Function
class BCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BCELoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute Binary Cross Entropy Loss
        bce_loss = torch.mean(F.binary_cross_entropy(preds, target, reduction="none"), dim=1)
        bce_loss = torch.mul(bce_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        bce_loss = torch.sum(bce_loss)

        return bce_loss


# Custom MSE Loss Function
class MSELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MSELoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute Binary Cross Entropy Loss
        mse_loss = torch.mean(F.mse_loss(preds, target, reduction="none"), dim=1)
        mse_loss = torch.mul(mse_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        mse_loss = torch.sum(mse_loss)

        return mse_loss


# Custom Binary Dice Loss Function
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of prediction and target. Shape = [BS, ]
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice loss of shape
        dice_loss = 1.0 - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice_loss = torch.sum(dice_loss)

        return dice_loss


# Custom DiceBCE Loss
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of prediction and target. Shape = [BS, ]
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice loss of shape
        dice_loss = 1.0 - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice_loss = torch.sum(dice_loss)

        # Calculate BCE
        BCE = torch.mean(F.binary_cross_entropy(preds, target, reduction="none"), dim=1)
        BCE = torch.mul(BCE, has_mask) / (torch.sum(has_mask) + self.smooth)
        BCE = torch.sum(BCE)

        # Calculate combined loss
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


# Custom Jaccard/IoU Loss Function
class IoU_Loss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoU_Loss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        total = torch.sum(preds, dim=1) + torch.sum(target, dim=1)

        # Compute the Union of the prediction and target
        union = total - intersection

        # Compute IoU Loss
        IoU = 1.0 - torch.divide((intersection + self.smooth), (union + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        IoU = torch.mul(IoU, has_mask) / (torch.sum(has_mask) + self.smooth)
        IoU = torch.sum(IoU)

        return IoU


# Custom Focal Loss for Segmentation
class FocalLoss(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    FocalLoss class implements the Focal Loss for binary classification tasks.

    Args:
        alpha (float): Weighting factor to balance positive and negative class samples.
                       If alpha is set to -1, no weighting is applied.
                       positive class = alpha, negative class = (1-alpha)
        gamma (float): Focusing parameter controlling the degree of emphasis on hard-to-classify samples.
        reduction (str): Specifies the reduction to apply to the computed loss.
                         Supported values are 'none', 'mean', and 'sum'.

    Attributes:
        alpha (float): Weighting factor for class balancing.
        gamma (float): Focusing parameter for the Focal Loss.
        reduction (str): Reduction mode for the computed loss.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
        __call__(self, preds, targets):
            Computes the Focal Loss based on the provided predictions and targets.

    Example:
        focal_loss = Focal_Cls(alpha=0.25, gamma=2, reduction='mean')
        loss = focal_loss(predictions, targets)
    """

    def __init__(self, alpha, gamma, reduction='mean', smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, targets, has_mask, labels_cls, batch_idx):
        """
        Compute Focal Loss based on the provided predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed Focal Loss based on the specified reduction mode.
        """

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == targets.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        # Compute Binary Cross Entropy Loss
        bce = F.binary_cross_entropy(preds, targets, reduction='none')

        # Compute Focal Loss
        p_t = preds * targets + (1 - preds) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)

        # Apply the alpha weighting factor
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        loss = torch.mean(loss, dim=1)
        loss = torch.mul(loss, has_mask) / (torch.sum(has_mask) + self.smooth)

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} "
                f"\n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        return loss


# Custom Weakly-Supervised Loss (Max. pixel value)
class DiceBCE_WeakSup_Loss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(DiceBCE_WeakSup_Loss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation and maximum pixel values
        preds = self.sigmoid(preds)
        max_val, _ = torch.max(preds.view(preds.shape[0], -1), dim=1, keepdim=True)

        """SUPERVISED LOSS"""
        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of prediction and target. Shape = [BS, ]
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice loss of shape
        dice_loss = 1.0 - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice_loss = torch.sum(dice_loss)

        # Calculate BCE
        BCE_sup = torch.mean(F.binary_cross_entropy(preds, target, reduction="none"), dim=1)
        BCE_sup = torch.mul(BCE_sup, has_mask) / (torch.sum(has_mask) + self.smooth)
        BCE_sup = torch.sum(BCE_sup)

        # Calculate combined loss
        dice_BCE_sup = BCE_sup + dice_loss

        """SEMI-SUPERVISED LOSS"""
        BCE_semsup = torch.mean(
            F.binary_cross_entropy(max_val, labels_cls, reduction="none"),
            dim=1,
        )
        BCE_semsup = torch.mean(BCE_semsup)

        """COMPUTE COMBINED LOSS"""
        comb_loss = dice_BCE_sup + BCE_semsup

        return comb_loss


# Custom DiceFocal Loss Function
class DiceFocalLoss(nn.Module):
    """
    Custom Loss function combining Dice Loss and Focal Loss
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    FocalLoss class implements the Focal Loss for binary classification tasks.

    Args:
        alpha (float): Weighting factor to balance positive and negative class samples.
                       If alpha is set to -1, no weighting is applied.
                       positive class = alpha, negative class = (1-alpha)
        gamma (float): Focusing parameter controlling the degree of emphasis on hard-to-classify samples.
        smooth (float, optional): Smoothing factor to prevent division by zero in Dice Loss. Default is 1e-6.
        p (int, optional): Exponent parameter for computation of Dice Loss. Default is 1.

    Attributes:
        alpha (float): Weighting factor for class balancing.
        gamma (float): Focusing parameter for the Focal Loss.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
        __call__(self, preds, targets):
            Computes the Dice+Focal Loss based on the provided predictions and targets.

    Returns:
        dice_focal (torch.Tensor): Combined Dice-Focal Loss.
    """

    def __init__(self, alpha, gamma, smooth=1e-6, p=1):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, targets, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == targets.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        """DICE LOSS"""
        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, targets), dim=1)

        # Compute the sum of prediction and target. Shape = [BS, ]
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(targets.pow(self.p), dim=1)

        # Compute Dice loss of shape
        dice_loss = 1.0 - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice_loss = torch.sum(dice_loss)

        """FOCAL LOSS"""
        # Compute Binary Cross Entropy Loss
        bce = F.binary_cross_entropy(preds, targets, reduction='none')

        # Compute Focal Loss
        p_t = preds * targets + (1 - preds) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)

        # Apply the alpha weighting factor
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        loss = torch.mean(loss, dim=1)
        loss = torch.mul(loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        loss = torch.sum(loss)

        """COMBINED LOSS"""
        dice_focal = dice_loss + loss

        return dice_focal


"""""" """""" """""" """""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM SEGMENTATION LOSS FUNCTIONS (Multi) """
"""""" """""" """""" """""" """""" """""" """""" """""" """"""


# Custom Multi-Mask BCE Loss Function
class MultiMaskBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MultiMaskBCELoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)

        # Initialize the BCE Loss
        bce_loss_complete = 0.0

        # Loop over the 4 masks and compute the BCE Loss
        for i in range(target.shape[1]):

            # Extract the target mask
            target_mask = target[:, i, :, :]

            # Extract the target mask and flatten it [BS, h*w]
            target_mask = target_mask.contiguous().view(target_mask.shape[0], -1)

            # Compute Binary Cross Entropy Loss
            bce_loss = torch.mean(F.binary_cross_entropy(preds, target_mask, reduction="none"), dim=1)
            bce_loss = torch.mul(bce_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
            bce_loss = torch.sum(bce_loss)

            # Accumulate the BCE Loss
            bce_loss_complete += (bce_loss/target.shape[1])

        return bce_loss_complete


# Custom Multi-Mask MSE Loss Function
class MultiMaskMSELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MultiMaskMSELoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)

        # Initialize the MSE Loss
        mse_loss_complete = 0.0

        # Loop over the 4 masks and compute the MSE Loss
        for i in range(target.shape[1]):

            # Extract the target mask
            target_mask = target[:, i, :, :]

            # Extract the target mask and flatten it [BS, h*w]
            target_mask = target_mask.contiguous().view(target_mask.shape[0], -1)

            # Compute Mean Squared Error Loss
            mse_loss = torch.mean(F.mse_loss(preds, target_mask, reduction="none"), dim=1)
            mse_loss = torch.mul(mse_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
            mse_loss = torch.sum(mse_loss)

            # Accumulate the MSE Loss
            mse_loss_complete += (mse_loss/target.shape[1])

        return mse_loss_complete


# Custom Multi-Mask DICE Loss Function
class MultiMaskDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1, variant='Regular'):
        super(MultiMaskDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()
        self.variant = variant

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction [BS, c, h, w] and target match [BS, 4c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)

        # Initialize the Dice Loss
        dice_loss_complete = 0.0

        # Loop over the 4 masks and compute the Dice Loss
        for i in range(target.shape[1]):

            # Extract the target mask
            target_mask = target[:, i, :, :]

            # Extract the target mask and flatten it [BS, h*w]
            target_mask = target_mask.contiguous().view(target_mask.shape[0], -1)

            # Compute intersection between prediction and target. Shape = [BS, ]
            intersection = torch.sum(torch.mul(preds, target_mask), dim=1)

            # Compute the sum of prediction and target. Shape = [BS, ]
            denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target_mask.pow(self.p), dim=1)

            # Compute Dice loss of shape
            dice_loss = 1.0 - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

            # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
            dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
            dice_loss = torch.sum(dice_loss)

            # Accumulate the Dice Loss
            if self.variant == 'Weighted':
                dice_loss_complete += ((dice_loss * (i + 1))/(sum(range(1, target.shape[1] + 1))))
            else:
                dice_loss_complete += (dice_loss/target.shape[1])

        return dice_loss_complete


# Custom Multi-Mask DICE-BCE Loss Function
class MultiMaskDiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1, variant='Regular'):
        super(MultiMaskDiceBCELoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()
        self.variant = variant

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction [BS, c, h, w] and target match [BS, 4c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)

        # Initialize the DiceBCE Loss
        dice_bce_loss_complete = 0.0

        # Loop over the 4 masks and compute the Dice Loss
        for i in range(target.shape[1]):

            # Extract the target mask
            target_mask = target[:, i, :, :]

            # Extract the target mask and flatten it [BS, h*w]
            target_mask = target_mask.contiguous().view(target_mask.shape[0], -1)

            # Compute intersection between prediction and target. Shape = [BS, ]
            intersection = torch.sum(torch.mul(preds, target_mask), dim=1)

            # Compute the sum of prediction and target. Shape = [BS, ]
            denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target_mask.pow(self.p), dim=1)

            # Compute Dice loss of shape
            dice_loss = 1.0 - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

            # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
            dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
            dice_loss = torch.sum(dice_loss)

            # Calculate BCE
            bce_loss = torch.mean(F.binary_cross_entropy(preds, target_mask, reduction="none"), dim=1)
            bce_loss = torch.mul(bce_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
            bce_loss = torch.sum(bce_loss)

            # Combine loss
            dice_bce = bce_loss + dice_loss

            # Combine and accumulate loss
            if self.variant == 'Weighted':
                dice_bce_loss_complete += ((dice_bce * (i + 1))/(sum(range(1, target.shape[1] + 1))))
            else:
                dice_bce_loss_complete += (dice_bce/target.shape[1])

        return dice_bce_loss_complete


# Custom Multi-Mask JIGS (Jaccard Index for Golden Standard ground truth) Loss Function
class JIGSLoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(JIGSLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)

        # Extract the soft and hard target masks
        soft_mask = target[:, 0, :, :]
        hard_mask = target[:, -1, :, :]

        # Flatten the targets. Shape = [BS, h*w]]
        soft_mask = soft_mask.contiguous().view(soft_mask.shape[0], -1)
        hard_mask = hard_mask.contiguous().view(hard_mask.shape[0], -1)

        # Compute intersection between prediction and soft spot mask
        intersection_soft = torch.sum(torch.mul(preds, soft_mask), dim=1)

        # Compute intersection union between prediction and hard spot mask
        intersection_hard = torch.sum(torch.mul(preds, hard_mask), dim=1)
        union_hard = torch.sum(preds, dim=1) + torch.sum(hard_mask, dim=1) - intersection_hard

        # Compute JIGS Loss
        jigs = 1.0 - torch.divide(intersection_soft, (union_hard + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        jigs = torch.mul(jigs, has_mask) / (torch.sum(has_mask) + self.smooth)
        jigs = torch.sum(jigs)

        return jigs


# Custom Multi-Mask JIGS-BCE (Jaccard Index for Golden Standard ground truth) Loss Function
class JIGSBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(JIGSBCELoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)

        # Extract the soft and hard target masks
        soft_mask = target[:, 0, :, :]
        hard_mask = target[:, -1, :, :]

        # Flatten the targets. Shape = [BS, h*w]]
        soft_mask = soft_mask.contiguous().view(soft_mask.shape[0], -1)
        hard_mask = hard_mask.contiguous().view(hard_mask.shape[0], -1)

        # Compute intersection between prediction and soft spot mask
        intersection_soft = torch.sum(torch.mul(preds, soft_mask), dim=1)

        # Compute intersection union between prediction and hard spot mask
        intersection_hard = torch.sum(torch.mul(preds, hard_mask), dim=1)
        union_hard = torch.sum(preds, dim=1) + torch.sum(hard_mask, dim=1) - intersection_hard

        # Compute JIGS Loss
        jigs = 1.0 - torch.divide(intersection_soft, (union_hard + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        jigs = torch.mul(jigs, has_mask) / (torch.sum(has_mask) + self.smooth)
        jigs = torch.sum(jigs)

        # Calculate BCE for soft and hard mask
        bce_s = torch.mean(F.binary_cross_entropy(preds, soft_mask, reduction="none"), dim=1)
        bce_h = torch.mean(F.binary_cross_entropy(preds, hard_mask, reduction="none"), dim=1)
        bce = (bce_s + bce_h) / 2
        bce = torch.mul(bce, has_mask) / (torch.sum(has_mask) + self.smooth)
        bce = torch.sum(bce)

        # Combine loss
        jigs_bce = jigs + bce

        return jigs_bce
