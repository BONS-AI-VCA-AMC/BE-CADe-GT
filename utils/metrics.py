"""IMPORT PACKAGES"""
import torch
from torch import nn

"""""" """""" """""" """""" """""" """""" """"""
"""" DEFINE HELPER FUNCTIONS FOR METRIC"""
"""""" """""" """""" """""" """""" """""" """"""


def construct_metric(opt):
    # Define possible choices
    if opt.seg_metric == 'Dice':
        metric = BinaryDiceMetric(smooth=1e-6, p=1)
    elif opt.seg_metric == 'MultiMaskDice':
        metric = MultiMaskBinaryDiceMetric(smooth=1e-6, p=1, threshold=0.5, variant='Regular')
    elif opt.seg_metric == 'MultiMaskDiceW':
        metric = MultiMaskBinaryDiceMetric(smooth=1e-6, p=1, threshold=0.5, variant='Weighted')
    else:
        raise Exception('Unexpected Metric {}'.format(opt.metric))

    return metric


"""""" """""" """""" """"""
"""" CUSTOM METRICS"""
"""""" """""" """""" """"""
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook


# Binary Dice Coefficient for Training (Batches)
class BinaryDiceMetric(nn.Module):
    def __init__(self, smooth=1e-6, p=1, threshold=0.5):
        super(BinaryDiceMetric, self).__init__()
        self.smooth = smooth
        self.p = p
        self.threshold = threshold
        self.dice_accumulator = list()

    def update(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction and target. Shape = [BS, c*h*w]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold
        target = target.contiguous().view(target.shape[0], -1) > self.threshold

        # Compute intersection between prediction and target
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice Coefficient
        dice = torch.divide((2 * intersection), (denominator + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice = torch.sum(dice)
        self.dice_accumulator.append(dice)

    def compute(self):
        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.mean(dice_accumulator)

        return avg_dice_score

    def reset(self):
        self.dice_accumulator = list()


# Binary Dice Coefficient for Evaluation (Single images)
class BinaryDiceMetricEval(nn.Module):
    def __init__(self, smooth=1e-6, p=1, threshold=0.5):
        super(BinaryDiceMetricEval, self).__init__()
        self.smooth = smooth
        self.p = p
        self.threshold = threshold
        self.dice_accumulator = list()
        self.has_mask_accumulator = 0

    def update(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction and target. Shape = [BS, c*h*w]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold
        target = target.contiguous().view(target.shape[0], -1) > self.threshold

        # Compute intersection between prediction and target
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice Coefficient
        dice = torch.divide((2 * intersection), (denominator + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice = torch.sum(dice)
        self.dice_accumulator.append(dice)
        self.has_mask_accumulator += torch.any(target)

    def compute(self):
        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.sum(dice_accumulator) / self.has_mask_accumulator

        return avg_dice_score

    def compute_single(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction and target. Shape = [BS, c*h*w]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold
        target = target.contiguous().view(target.shape[0], -1) > self.threshold

        # Compute intersection between prediction and target
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice Coefficient
        dice = torch.divide((2 * intersection), (denominator + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice = torch.sum(dice)

        return dice

    def reset(self):
        self.dice_accumulator = list()
        self.has_mask_accumulator = 0


# Custom Multi-Mask DICE Metric for Training (Batches)
class MultiMaskBinaryDiceMetric(nn.Module):
    def __init__(self, smooth=1e-6, p=1, threshold=0.5, variant='Regular'):
        super(MultiMaskBinaryDiceMetric, self).__init__()
        self.smooth = smooth
        self.p = p
        self.threshold = threshold
        self.dice_accumulator = list()
        self.variant = variant

    def update(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold

        # Initialize the dice score accumulator for different masks
        dice_score = 0.0

        # Loop over the masks
        for i in range(target.shape[1]):

            # Extract the target mask
            target_mask = target[:, i, :, :]
            target_mask = target_mask.contiguous().view(target_mask.shape[0], -1) > self.threshold

            # Compute intersection between prediction and target
            intersection = torch.sum(torch.mul(preds, target_mask), dim=1)

            # Compute the sum of predictions and target
            denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target_mask.pow(self.p), dim=1)

            # Compute Dice Coefficient
            dice = torch.divide((2 * intersection), (denominator + self.smooth))

            # Multiply with has_mask to only have coefficient for samples with mask
            dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
            dice = torch.sum(dice)

            # Average the Dice Loss
            if self.variant == 'Weighted':
                dice_score += ((dice * (i + 1))/(sum(range(1, target.shape[1] + 1))))
            else:
                dice_score += (dice/target.shape[1])

        # Update the dice accumulator
        self.dice_accumulator.append(dice_score)

    def compute(self):
        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.mean(dice_accumulator)

        return avg_dice_score

    def reset(self):
        self.dice_accumulator = list()


# Custom Multi-Mask DICE Metric for Evaluation (Single images)
class MultiMaskBinaryDiceMetricEval(nn.Module):
    def __init__(self, smooth=1e-6, p=1, threshold=0.5, variant='Regular'):
        super(MultiMaskBinaryDiceMetricEval, self).__init__()
        self.smooth = smooth
        self.p = p
        self.threshold = threshold
        self.dice_accumulator = list()
        self.has_mask_accumulator = 0
        self.variant = variant

    def update(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold

        # Initialize the dice score accumulator for different masks
        dice_score = 0.0

        # Loop over the masks
        for i in range(target.shape[1]):

            # Extract the target mask
            target_mask = target[:, i, :, :]
            target_mask = target_mask.contiguous().view(target_mask.shape[0], -1) > self.threshold

            # Compute intersection between prediction and target
            intersection = torch.sum(torch.mul(preds, target_mask), dim=1)

            # Compute the sum of predictions and target
            denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target_mask.pow(self.p), dim=1)

            # Compute Dice Coefficient
            dice = torch.divide((2 * intersection), (denominator + self.smooth))

            # Multiply with has_mask to only have coefficient for samples with mask
            dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
            dice = torch.sum(dice)

            # Average the Dice Loss
            if self.variant == 'Weighted':
                dice_score += ((dice * (i + 1))/(sum(range(1, target.shape[1] + 1))))
            else:
                dice_score += (dice/target.shape[1])

        # Update the dice accumulator
        self.dice_accumulator.append(dice_score)
        self.has_mask_accumulator += torch.any(target)

    def compute(self):
        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.sum(dice_accumulator) / self.has_mask_accumulator

        return avg_dice_score

    def compute_single(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold

        # Initialize the dice score accumulator for different masks
        dice_score = 0.0

        # Loop over the masks
        for i in range(target.shape[1]):

            # Extract the target mask
            target_mask = target[:, i, :, :]
            target_mask = target_mask.contiguous().view(target_mask.shape[0], -1) > self.threshold

            # Compute intersection between prediction and target
            intersection = torch.sum(torch.mul(preds, target_mask), dim=1)

            # Compute the sum of predictions and target
            denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target_mask.pow(self.p), dim=1)

            # Compute Dice Coefficient
            dice = torch.divide((2 * intersection), (denominator + self.smooth))

            # Multiply with has_mask to only have coefficient for samples with mask
            dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
            dice = torch.sum(dice)

            # Average the Dice Loss
            if self.variant == 'Weighted':
                dice_score += ((dice * (i + 1)) / (sum(range(1, target.shape[1] + 1))))
            else:
                dice_score += (dice / target.shape[1])

        return dice_score

    def reset(self):
        self.dice_accumulator = list()
        self.has_mask_accumulator = 0
