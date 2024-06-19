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
    elif opt.seg_metric == 'IoU':
        metric = IoU_Metric(smooth=1e-6)
    elif opt.seg_metric == 'JIGS':
        metric = JIGSMetric(smooth=1e-6, threshold=0.5)
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


# Jaccard Index / Intersection over Union
class IoU_Metric(nn.Module):
    def __init__(self, smooth=1e-6, threshold=0.5):
        super(IoU_Metric, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
        self.IoU_accumulator = list()

    def update(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction and target. Shape = [BS, c*h*w]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold
        target = target.contiguous().view(target.shape[0], -1) > self.threshold

        # Compute intersection between prediction and target
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        total = torch.sum(preds, dim=1) + torch.sum(target, dim=1)

        # Compute the Union of the prediction and target
        union = total - intersection

        # Compute IoU
        IoU = torch.divide((intersection + self.smooth), (union + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        IoU = torch.mul(IoU, has_mask) / (torch.sum(has_mask) + self.smooth)
        IoU = torch.sum(IoU)
        self.IoU_accumulator.append(IoU)

    def compute(self):
        # Convert list to tensor and compute mean
        IoU_accumulator = torch.FloatTensor(self.IoU_accumulator)
        avg_IoU_score = torch.mean(IoU_accumulator)

        return avg_IoU_score

    def reset(self):
        self.IoU_accumulator = list()


# JIGS Metric (Jaccard Index for Golden Standard ground truth) for Training (Batches)
class JIGSMetric(nn.Module):
    def __init__(self, smooth=1e-6, threshold=0.5):
        super(JIGSMetric, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
        self.jigs_accumulator = list()

    def update(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & soft spot shape don't match"

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold

        # Extract the soft and hard target masks
        soft_mask = target[:, 0, :, :]
        hard_mask = target[:, -1, :, :]

        # Flatten the targets. Shape = [BS, h*w]]
        soft_mask = soft_mask.contiguous().view(soft_mask.shape[0], -1) > self.threshold
        hard_mask = hard_mask.contiguous().view(hard_mask.shape[0], -1) > self.threshold

        # Compute intersection between prediction and soft spot mask
        intersection_soft = torch.sum(torch.mul(preds, soft_mask), dim=1)

        # Compute intersection and union between prediction and hard spot mask
        union_hard = torch.sum(torch.logical_or(preds, hard_mask), dim=1)
        # intersection_hard = torch.sum(torch.mul(preds, hard_mask), dim=1)
        # union_hard = torch.sum(preds, dim=1) + torch.sum(hard_mask, dim=1) - intersection_hard

        # Compute JIGS
        jigs = torch.divide(intersection_soft, (union_hard + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        jigs = torch.mul(jigs, has_mask) / (torch.sum(has_mask) + self.smooth)
        jigs = torch.sum(jigs)
        self.jigs_accumulator.append(jigs)

    def compute(self):
        # Convert list to tensor and compute mean
        jigs_accumulator = torch.FloatTensor(self.jigs_accumulator)
        avg_jigs_score = torch.mean(jigs_accumulator)

        return avg_jigs_score

    def reset(self):
        self.jigs_accumulator = list()


# JIGS Metric (Jaccard Index for Golden Standard ground truth) for Evaluation (Single images)
class JIGSMetricEval(nn.Module):
    def __init__(self, smooth=1e-6, threshold=0.5):
        super(JIGSMetricEval, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
        self.jigs_accumulator = list()
        self.has_mask_accumulator = 0

    def update(self, preds, target, has_mask):
        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & soft spot shape don't match"

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold

        # Extract the soft and hard target masks
        soft_mask = target[:, 0, :, :]
        hard_mask = target[:, -1, :, :]

        # Flatten the targets. Shape = [BS, h*w]]
        soft_mask = soft_mask.contiguous().view(soft_mask.shape[0], -1) > self.threshold
        hard_mask = hard_mask.contiguous().view(hard_mask.shape[0], -1) > self.threshold

        # Compute intersection between prediction and soft spot mask
        intersection_soft = torch.sum(torch.mul(preds, soft_mask), dim=1)

        # Compute intersection and union between prediction and hard spot mask
        union_hard = torch.sum(torch.logical_or(preds, hard_mask), dim=1)
        # intersection_hard = torch.sum(torch.mul(preds, hard_mask), dim=1)
        # union_hard = torch.sum(preds, dim=1) + torch.sum(hard_mask, dim=1) - intersection_hard

        # Compute JIGS
        jigs = torch.divide(intersection_soft, (union_hard + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        jigs = torch.mul(jigs, has_mask) / (torch.sum(has_mask) + self.smooth)
        jigs = torch.sum(jigs)
        self.jigs_accumulator.append(jigs)
        self.has_mask_accumulator += torch.any(soft_mask)

    def compute_single(self, preds, target, has_mask):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & soft spot shape don't match"

        # Flatten the prediction. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1) > self.threshold

        # Extract the soft and hard target masks
        soft_mask = target[:, 0, :, :]
        hard_mask = target[:, -1, :, :]

        # Flatten the targets. Shape = [BS, h*w]]
        soft_mask = soft_mask.contiguous().view(soft_mask.shape[0], -1) > self.threshold
        hard_mask = hard_mask.contiguous().view(hard_mask.shape[0], -1) > self.threshold

        # Compute intersection between prediction and soft spot mask
        intersection_soft = torch.sum(torch.mul(preds, soft_mask), dim=1)

        # Compute intersection and union between prediction and hard spot mask
        # union_hard = torch.sum(torch.logical_or(preds, hard_mask), dim=1)
        intersection_hard = torch.sum(torch.mul(preds, hard_mask), dim=1)
        union_hard = torch.sum(preds, dim=1) + torch.sum(hard_mask, dim=1) - intersection_hard

        # Compute JIGS
        jigs = torch.divide(intersection_soft, (union_hard + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        jigs = torch.mul(jigs, has_mask) / (torch.sum(has_mask) + self.smooth)
        jigs = torch.sum(jigs)

        return jigs

    def compute(self):
        # Convert list to tensor and compute mean
        jigs_accumulator = torch.FloatTensor(self.jigs_accumulator)
        avg_jigs_score = torch.sum(jigs_accumulator) / self.has_mask_accumulator

        return avg_jigs_score

    def reset(self):
        self.jigs_accumulator = list()
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


# Interobserver Binary Dice Metric
class InterObserverBinaryDiceMetric(nn.Module):
    def __init__(self, smooth=1e-6, p=1, label_smooth=0.01):
        super(InterObserverBinaryDiceMetric, self).__init__()
        self.smooth = smooth
        self.p = p
        self.dice_accumulator = list()
        self.margin = label_smooth * 0.5

    def update(self, preds, targets, has_mask):
        # Check whether the batch sizes of prediction adn target match
        assert preds.shape[0] == targets.shape[0], "pred & target batch size don't match"

        # Loop over the batch size
        for i in range(preds.shape[0]):
            # Specify predictions [X, 256, 256] and target [X, 256, 256]
            pred = preds[i, :, :, :]
            pred = torch.cat([pred] * targets.shape[1], dim=0)
            target = torch.round(targets[i, :, :, :], decimals=2)

            # Specify has_mask [X] and initialize weights [X]
            hm = has_mask[i, :] if preds.shape[0] > 1 else has_mask
            weights = torch.zeros([target.shape[0]])

            # Find number of unique values in the target masks without 0.0
            unique_values = torch.unique(target).tolist()
            if 0 in unique_values:
                unique_values.remove(0)

            # Threshold the predictions according to the unique values in the mask and reshape
            if len(unique_values) > 0:
                # # Plot unique values plot
                # if len(unique_values) == 1:
                #     fig, axs = plt.subplots(2)
                #     axs[0].imshow(pred[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[0].set_title('Max: {}'.format(round(torch.max(pred[0, :, :]).item(), 4)))
                #     axs[0].axis('off')
                #     axs[1].imshow(target[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[1].set_title('Max: {}'.format(round(torch.max(target[0, :, :]).item(), 4)))
                #     axs[1].axis('off')
                #     fig.suptitle('Non-Thresholded')
                #     # plt.tight_layout()
                #     plt.show()
                # else:
                #     fig, axs = plt.subplots(2, len(unique_values))
                #     for y in range(len(unique_values)):
                #         axs[0, y].imshow(pred[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[0, y].set_title('Max: {}'.format(round(torch.max(pred[y, :, :]).item(), 4)))
                #         axs[0, y].axis('off')
                #         axs[1, y].imshow(target[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[1, y].set_title('Max: {}'.format(round(torch.max(target[y, :, :]).item(), 4)))
                #         axs[1, y].axis('off')
                #     fig.suptitle('Non-Thresholded')
                #     # plt.tight_layout()
                #     plt.show()

                for j in range(len(unique_values)):
                    pred[j, :, :] = pred[j, :, :] >= unique_values[j]
                    target[j, :, :] = target[j, :, :] >= unique_values[j]
                    weights[j] = unique_values[j]

                # # Plot unique values plot
                # if len(unique_values) == 1:
                #     fig, axs = plt.subplots(2)
                #     axs[0].imshow(pred[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[0].set_title('Max: {}'.format(round(torch.max(pred[0, :, :]).item(), 4)))
                #     axs[0].axis('off')
                #     axs[1].imshow(target[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[1].set_title('Max: {}'.format(round(torch.max(target[0, :, :]).item(), 4)))
                #     axs[1].axis('off')
                #     fig.suptitle('Thresholded')
                #     # plt.tight_layout()
                #     plt.show()
                # else:
                #     fig, axs = plt.subplots(2, len(unique_values))
                #     for y in range(len(unique_values)):
                #         axs[0, y].imshow(pred[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[0, y].set_title('Max: {}'.format(round(torch.max(pred[y, :, :]).item(), 4)))
                #         axs[0, y].axis('off')
                #         axs[1, y].imshow(target[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[1, y].set_title('Max: {}'.format(round(torch.max(target[y, :, :]).item(), 4)))
                #         axs[1, y].axis('off')
                #     fig.suptitle('Thresholded')
                #     # plt.tight_layout()
                #     plt.show()

                # Flatten the prediction and target
                pred = pred.contiguous().view(pred.shape[0], -1)
                target = target.contiguous().view(target.shape[0], -1)

                # Compute intersection between prediction and target
                intersection = torch.sum(torch.mul(pred, target), dim=1)

                # Compute the sum of predictions and target
                denominator = torch.sum(pred.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

                # Compute Dice Coefficient
                dice = torch.divide((2 * intersection), (denominator + self.smooth))

                # Optionally weight the individual dice scores
                # dice = torch.mul(dice, weights)

                # Multiply with has_mask to only have coefficient for samples with mask
                dice = torch.mul(dice, hm) / (torch.sum(hm) + self.smooth)
                dice = torch.sum(dice)

                # Update dice total accumulator
                self.dice_accumulator.append(dice)

                return dice
        else:
            return 0

    def compute(self):
        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.mean(dice_accumulator)

        return avg_dice_score

    def reset(self):
        self.dice_accumulator = list()
