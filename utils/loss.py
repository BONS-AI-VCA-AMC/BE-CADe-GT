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
    else:
        raise Exception("Unexpected Classification Loss {}".format(opt.cls_criterion))

    # Define possible choices for segmentation loss (Single Mask)
    if opt.seg_criterion == "Dice":
        seg_criterion = BinaryDiceLoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == "DiceBCE":
        seg_criterion = DiceBCELoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == 'MSE':
        seg_criterion = MSELoss(smooth=1e-6)
    elif opt.seg_criterion == 'BCE':
        seg_criterion = BCELoss(smooth=1e-6)

    # Define possible choices for segmentation loss (Multi Mask)
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

