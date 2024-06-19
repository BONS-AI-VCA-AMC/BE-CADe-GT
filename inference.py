"""IMPORT PACKAGES"""
import os
import argparse
import time
import json
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from torchinfo import summary
from sklearn.metrics import roc_curve, roc_auc_score

from data.dataset import read_inclusion, augmentations
from train import check_cuda, find_best_model
from models.model import Model
import matplotlib.pyplot as plt

"""""" """""" """""" """"""
"""" HELPER FUNCTIONS """
"""""" """""" """""" """"""


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    criteria['dev'] = {
        'modality': ['wle'],
        'dataset': ['validation'],
        'min_height': None,
        'min_width': None,
    }

    criteria['test'] = {
        'modality': ['wle'],
        'dataset': ['test'],
        'min_height': None,
        'min_width': None,
    }

    return criteria


# Define custom argument type for a list of enhancement settings
def list_of_settings(arg):
    return list(map(str, arg.split(',')))


# Define function for extracting masks
def extract_masks(image, masklist):
    # Create dictionary for masks
    mask_dict = {'Soft': 0, 'Plausible': 0, 'Sweet': 0, 'Hard': 0}

    # Extract information on expert
    expert_list = list(set([os.path.split(os.path.split(masklist[i])[0])[1] for i in range(len(masklist))]))
    expert_list.sort()

    # Set Bools for all masks
    lower0, higher0, lower1, higher1 = False, False, False, False
    ll0, hl0, ll1, hl1 = 0, 0, 0, 0

    # Loop over all masks
    for i in range(len(masklist)):
        # Extract information on expert and likelihood
        expert = os.path.split(os.path.split(masklist[i])[0])[1]
        likelihood = os.path.split(os.path.split(os.path.split(masklist[i])[0])[0])[1]

        # If ll0 mask is present
        if expert_list.index(expert) == 0 and 'Lower' in likelihood:
            lower0 = True
            ll0 = Image.open(masklist[i]).convert('1')
            if ll0.size != image.size:
                ll0 = np.array(ll0.resize(image.size, resample=Image.NEAREST))
            else:
                ll0 = np.array(ll0)

        # If hl0 mask is present
        elif expert_list.index(expert) == 0 and 'Higher' in likelihood:
            hl0 = Image.open(masklist[i]).convert('1')
            higher0 = True
            if hl0.size != image.size:
                hl0 = np.array(hl0.resize(image.size, resample=Image.NEAREST))
            else:
                hl0 = np.array(hl0)

        # If ll1 mask is present
        elif expert_list.index(expert) == 1 and 'Lower' in likelihood:
            ll1 = Image.open(masklist[i]).convert('1')
            lower1 = True
            if ll1.size != image.size:
                ll1 = np.array(ll1.resize(image.size, resample=Image.NEAREST))
            else:
                ll1 = np.array(ll1)

        # If hl1 mask is present
        elif expert_list.index(expert) == 1 and 'Higher' in likelihood:
            hl1 = Image.open(masklist[i]).convert('1')
            higher1 = True
            if hl1.size != image.size:
                hl1 = np.array(hl1.resize(image.size, resample=Image.NEAREST))
            else:
                hl1 = np.array(hl1)

        # If more than 2 experts are available, raise an error
        else:
            raise ValueError('More than 2 experts...')

    # Replace LL with LL U HL if they both exist to enforce the protocol
    if lower0 and higher0:
        ll0 = np.add(ll0, hl0)
    if lower1 and higher1:
        ll1 = np.add(ll1, hl1)

    """Create Consensus masks for each likelihood"""
    # Construct LowerLikelihood building blocks
    if lower0 + lower1 == 2:
        union_ll = np.add(ll0, ll1)
        intersection_ll = np.multiply(ll0, ll1)
    elif lower0 + lower1 == 1:
        if lower0:
            union_ll = ll0
            intersection_ll = ll0
        else:
            union_ll = ll1
            intersection_ll = ll1
    else:
        union_ll = 0
        intersection_ll = 0

    # Construct HigherLikelihood building blocks
    if higher0 + higher1 == 2:
        union_hl = np.add(hl0, hl1)
        intersection_hl = np.multiply(hl0, hl1)
    elif higher0 + higher1 == 1:
        if higher0:
            union_hl = hl0
            intersection_hl = hl0
        else:
            union_hl = hl1
            intersection_hl = hl1
    else:
        union_hl = 0
        intersection_hl = 0

    # Construct consensus masks
    if lower0 + lower1 == 0:
        soft = Image.fromarray(union_hl).convert('1')
        plausible = Image.fromarray(union_hl).convert('1')
        sweet = Image.fromarray(union_hl).convert('1')
        hard = Image.fromarray(intersection_hl).convert('1')
    elif higher0 + higher1 == 0:
        soft = Image.fromarray(union_ll).convert('1')
        plausible = Image.fromarray(intersection_ll).convert('1')
        sweet = Image.fromarray(intersection_ll).convert('1')
        hard = Image.fromarray(intersection_ll).convert('1')
    elif lower0 + lower1 == 1:
        soft = Image.fromarray(np.add(intersection_ll, union_hl)).convert('1')
        plausible = Image.fromarray(np.add(intersection_ll, union_hl)).convert('1')
        sweet = Image.fromarray(union_hl).convert('1')
        hard = Image.fromarray(intersection_hl).convert('1')
    else:
        soft = Image.fromarray(union_ll).convert('1')
        plausible = Image.fromarray(np.add(intersection_ll, union_hl)).convert('1')
        sweet = Image.fromarray(union_hl).convert('1')
        hard = Image.fromarray(intersection_hl).convert('1')

    # Store in dictionary
    mask_dict['Soft'] = soft
    mask_dict['Plausible'] = plausible
    mask_dict['Sweet'] = sweet
    mask_dict['Hard'] = hard

    return mask_dict


# Create function for creating biopsy with radius
def create_biopsy(mask, biopsy, radius):
    """
    Note: works only on full images, not on cropped images. Correct for this in the future
    """

    # Create an empty array of same size as mask
    mask_copy = np.zeros_like(np.array(mask))

    # Create biopsy with radius
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i ** 2 + j ** 2 <= radius ** 2:
                if 0 <= biopsy[1] + i < mask.shape[0] and 0 <= biopsy[0] + j < mask.shape[1]:
                    mask_copy[biopsy[1] + i, biopsy[0] + j] = 255

    mask_copy = np.array(Image.fromarray(mask_copy).convert('1'))

    return mask_copy


"""""" """""" """""" """""" """"""
"""" FUNCTIONS FOR INFERENCE """
"""""" """""" """""" """""" """"""


def run_val(opt, exp_name):

    # Test Device
    device = check_cuda()

    # Print info line
    print('Performing threshold analysis on validation set...')

    # Construct data
    criteria = get_data_inclusion_criteria()
    val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['dev'])
    print('Found {} images...'.format(len(val_inclusion)))

    # Construct transforms
    data_transforms = augmentations(opt=opt)

    # Construct Model and load weights
    model = Model(opt=opt)
    best_index = find_best_model(path=os.path.join(SAVE_DIR, exp_name))
    checkpoint = torch.load(os.path.join(SAVE_DIR, exp_name, best_index))['state_dict']

    # Adapt state_dict keys (remove model. from the key and save again)
    if not os.path.exists(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt')):
        checkpoint_keys = list(checkpoint.keys())
        for key in checkpoint_keys:
            checkpoint[key.replace('model.', '')] = checkpoint[key]
            del checkpoint[key]
        model.load_state_dict(checkpoint, strict=True)
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'),
        )

    # Load weights
    weights = torch.load(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'))
    model.load_state_dict(weights, strict=True)

    # Initialize text file for thresholds
    f_val = open(os.path.join(SAVE_DIR, exp_name, 'thresholds.txt'), 'x')
    f_txt_val = open(os.path.join(SAVE_DIR, exp_name, 'thresholds.txt'), 'a')

    # Initialize metrics for classification
    y_true, y_pred, y_mask_pred = list(), list(), list()

    # Push model to GPU and set in evaluation mode
    model.cuda()
    model.eval()
    with torch.no_grad():
        # Loop over the data
        for img in val_inclusion:
            # Extract information from cache
            file = img['file']
            img_name = os.path.splitext(os.path.split(file)[1])[0]
            roi = img['roi']

            # Construct target
            label = img['label']
            if label:
                target = True
                y_true.append(target)
            else:
                target = False
                y_true.append(target)

            # Open Image
            image = Image.open(file).convert('RGB')

            # Crop the image to the ROI
            image = image.crop((roi[2], roi[0], roi[3], roi[1]))

            # Apply transforms to image and mask
            image_t, _ = data_transforms['test'](image, image, 0)
            image_t = image_t.unsqueeze(0).cuda()

            # Get prediction of model and perform Sigmoid activation
            out1, out2 = model(image_t)
            cls_pred = out1 if out1.dim() == 2 else out2
            seg_pred = out2 if out2.dim() == 4 else out1
            cls_pred = torch.sigmoid(cls_pred).cpu()
            seg_pred = torch.sigmoid(seg_pred).cpu()

            # Process segmentation prediction; positive prediction if 1 pixel exceeds threshold = 0.5
            mask = seg_pred.squeeze(axis=0)
            mask_cls_logit = torch.max(mask)

            # Append values to list
            y_pred.append(cls_pred.item())
            y_mask_pred.append(mask_cls_logit.item())

    # Compute AUC for classification (classification head)
    auc = roc_auc_score(y_true, y_pred)
    print('auc_cls: {:.4f}'.format(auc))
    fpr_cls, tpr_cls, thr_cls = roc_curve(y_true, y_pred)

    # Write Classification thresholds
    combined_score_cls_thr, combined_score_cls = list(), list()
    f_txt_val.write('### Thresholds (Classification Head)  ###')
    for i in range(len(tpr_cls)):
        if tpr_cls[i] >= opt.sens_val:
            f_txt_val.write('\nThreshold: {:.4f} | Sensitivity: {:.4f} | Specificity: {:.4f} | Combined: {:.4f}'.format(thr_cls[i], tpr_cls[i], 1-fpr_cls[i], tpr_cls[i] * (1-fpr_cls[i])))
            combined_score_cls_thr.append(thr_cls[i])
            combined_score_cls.append(tpr_cls[i] * (1-fpr_cls[i]))

    # Find maximal value
    max_score_cls = max(combined_score_cls)
    max_score_cls_thr = combined_score_cls_thr[combined_score_cls.index(max_score_cls)]
    f_txt_val.write('\n\nMaximal Combined Score: {:.4f} | Threshold: {:.4f}'.format(max_score_cls, max_score_cls_thr))

    # Compute AUC for segmentation
    auc = roc_auc_score(y_true, y_mask_pred)
    print('auc_seg: {:.4f}'.format(auc))
    fpr_seg, tpr_seg, thr_seg = roc_curve(y_true, y_mask_pred)

    # Write Segmentation thresholds
    combined_score_seg_thr, combined_score_seg = list(), list()
    f_txt_val.write('\n\n### Thresholds (Segmentation Head)  ###')
    for i in range(len(tpr_seg)):
        if tpr_seg[i] >= opt.sens_val:
            f_txt_val.write('\nThreshold: {:.4f} | Sensitivity: {:.4f} | Specificity: {:.4f} | Combined: {:.4f}'.format(thr_seg[i], tpr_seg[i], 1-fpr_seg[i], tpr_seg[i] * (1-fpr_seg[i])))
            combined_score_seg_thr.append(thr_seg[i])
            combined_score_seg.append(tpr_seg[i] * (1-fpr_seg[i]))

    # Find maximal value
    max_score_seg = max(combined_score_seg)
    max_score_seg_thr = combined_score_seg_thr[combined_score_seg.index(max_score_seg)]
    f_txt_val.write('\n\nMaximal Combined Score: {:.4f} | Threshold: {:.4f}'.format(max_score_seg, max_score_seg_thr))

    return max_score_cls_thr-0.0001, max_score_seg_thr-0.0001


def run(opt, f_txt, exp_name, inf_set, thr_cls, thr_seg):
    # Test Device
    device = check_cuda()

    # Print info line
    print(f'Performing analysis on {inf_set} set...')

    # Create model output database
    df = pd.DataFrame(columns=['Case', 'CLS', 'SEG', 'CLS Correct', 'SEG Correct', 'LOC (Soft)', 'LOC (Plaus)', 'LOC (Sweet)', 'LOC (Hard)'])
    logi = 0

    # Construct data
    criteria = get_data_inclusion_criteria()
    if inf_set == 'Test':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['test'])
        print('Found {} images...'.format(len(val_inclusion)))
    else:
        raise Exception('Unrecognized DEFINE_SET: {}'.format(inf_set))

    # Construct transforms
    data_transforms = augmentations(opt=opt)

    # Construct Model and load weights
    model = Model(opt=opt)
    best_index = find_best_model(path=os.path.join(SAVE_DIR, exp_name))
    checkpoint = torch.load(os.path.join(SAVE_DIR, exp_name, best_index))['state_dict']

    # Adapt state_dict keys (remove model. from the key and save again)
    if not os.path.exists(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt')):
        checkpoint_keys = list(checkpoint.keys())
        for key in checkpoint_keys:
            checkpoint[key.replace('model.', '')] = checkpoint[key]
            del checkpoint[key]
        model.load_state_dict(checkpoint, strict=True)
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'),
        )

    # Load weights
    weights = torch.load(os.path.join(SAVE_DIR, exp_name, 'final_pytorch_model.pt'))
    model.load_state_dict(weights, strict=True)

    # Initialize metrics for classification
    tp_cls, tn_cls, fp_cls, fn_cls = 0.0, 0.0, 0.0, 0.0
    tp_seg, tn_seg, fp_seg, fn_seg = 0.0, 0.0, 0.0, 0.0
    y_true, y_pred, y_mask_pred = list(), list(), list()

    # Initialize metrics for localization
    loc_soft_cls, loc_plaus_cls, loc_sweet_cls, loc_hard_cls = 0.0, 0.0, 0.0, 0.0
    loc_soft_seg, loc_plaus_seg, loc_sweet_seg, loc_hard_seg = 0.0, 0.0, 0.0, 0.0
    loc_total_cls, loc_total_seg = 0.0, 0.0

    # Initialize metrics for detection (combined classification (classification head) and localization)
    tp_det_soft_cls, tn_det_soft_cls, fp_det_soft_cls, fn_det_soft_cls = 0.0, 0.0, 0.0, 0.0
    tp_det_plaus_cls, tn_det_plaus_cls, fp_det_plaus_cls, fn_det_plaus_cls = 0.0, 0.0, 0.0, 0.0
    tp_det_sweet_cls, tn_det_sweet_cls, fp_det_sweet_cls, fn_det_sweet_cls = 0.0, 0.0, 0.0, 0.0
    tp_det_hard_cls, tn_det_hard_cls, fp_det_hard_cls, fn_det_hard_cls = 0.0, 0.0, 0.0, 0.0

    # Initialize metrics for detection (combined classification (segmentation head) and localization)
    tp_det_soft_seg, tn_det_soft_seg, fp_det_soft_seg, fn_det_soft_seg = 0.0, 0.0, 0.0, 0.0
    tp_det_plaus_seg, tn_det_plaus_seg, fp_det_plaus_seg, fn_det_plaus_seg = 0.0, 0.0, 0.0, 0.0
    tp_det_sweet_seg, tn_det_sweet_seg, fp_det_sweet_seg, fn_det_sweet_seg = 0.0, 0.0, 0.0, 0.0
    tp_det_hard_seg, tn_det_hard_seg, fp_det_hard_seg, fn_det_hard_seg = 0.0, 0.0, 0.0, 0.0

    # Load biopsy locations from experts
    with open(os.path.join('expert annotations', 'biopsy_locations.json'), 'r') as f:
        biopsy_locations = json.load(f)

    # Push model to GPU and set in evaluation mode
    model.cuda()
    model.eval()
    with torch.no_grad():
        # Loop over the data
        for img in val_inclusion:
            # Extract information from cache
            file = img['file']
            img_name = os.path.splitext(os.path.split(file)[1])[0]
            roi = img['roi']
            masklist = img['mask']

            # Construct target
            label = img['label']
            if label:
                target = True
                y_true.append(target)
            else:
                target = False
                y_true.append(target)

            # Process biopsy locations for neoplastic imagery
            if target:
                biopsy_experts = biopsy_locations[img_name]['biopsy']
                for loc in biopsy_experts:
                    loc[0] -= roi[2]
                    loc[1] -= roi[0]
                    loc[0] = int(loc[0] * opt.imagesize / (roi[3] - roi[2]))
                    loc[1] = int(loc[1] * opt.imagesize / (roi[1] - roi[0]))

            # Open Image
            image = Image.open(file).convert('RGB')

            # By default set has_mask to zero
            has_mask = False

            # Extract masks
            if len(masklist) > 0:
                mask_dict = extract_masks(image, masklist)
                mask_soft = np.array(mask_dict['Soft'].crop((roi[2], roi[0], roi[3], roi[1])).resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST))
                mask_plaus = np.array(mask_dict['Plausible'].crop((roi[2], roi[0], roi[3], roi[1])).resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST))
                mask_sweet = np.array(mask_dict['Sweet'].crop((roi[2], roi[0], roi[3], roi[1])).resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST))
                mask_hard = np.array(mask_dict['Hard'].crop((roi[2], roi[0], roi[3], roi[1])).resize((opt.imagesize, opt.imagesize), resample=Image.NEAREST))
                has_mask = True

            # Crop the image to the ROI
            image = image.crop((roi[2], roi[0], roi[3], roi[1]))

            # Apply transforms to image and mask
            image_t, _ = data_transforms['test'](image, image, has_mask)
            image_t = image_t.unsqueeze(0).cuda()

            # Get prediction of model and perform Sigmoid activation
            out1, out2 = model(image_t)
            cls_pred = out1 if out1.dim() == 2 else out2
            seg_pred = out2 if out2.dim() == 4 else out1
            cls_pred = torch.sigmoid(cls_pred).cpu()
            seg_pred = torch.sigmoid(seg_pred).cpu()

            # Process classification prediction; positive prediction if exceed threshold = 0.5
            cls = cls_pred > thr_cls
            cls = cls.squeeze(axis=0).item()

            # Process segmentation prediction; positive prediction if 1 pixel exceeds threshold = 0.5
            mask = seg_pred.squeeze(axis=0)
            mask_cls_logit = torch.max(mask)
            mask_cls = (torch.max(mask) > thr_seg).item()

            # Find the location of maximum value in the mask
            max_indices = (mask.squeeze(axis=0) == mask_cls_logit.item()).nonzero(as_tuple=False)
            if max_indices.shape[0] > 1:
                print(f'Multiple biopsy sites found ({max_indices.shape[0]}), picking random one...')
                index = random.randint(0, max_indices.shape[0] - 1)
                max_indices = [max_indices[index][1].item(), max_indices[index][0].item()]
            else:
                max_indices = [max_indices[0][1].item(), max_indices[0][0].item()]

            # Create biopsy with radius
            if len(masklist) > 0:
                new_radius_w = int(opt.radius * opt.imagesize / (roi[3] - roi[2]))
                new_radius_h = int(opt.radius * opt.imagesize / (roi[1] - roi[0]))
                new_radius = max(new_radius_w, new_radius_h)
                biopsy = create_biopsy(mask_soft, max_indices, new_radius)

            # Append values to list
            y_pred.append(cls_pred.item())
            y_mask_pred.append(mask_cls_logit.item())

            # Update classification metrics (classification head)
            tp_cls += target * cls
            tn_cls += (1 - target) * (1 - cls)
            fp_cls += (1 - target) * cls
            fn_cls += target * (1 - cls)

            # Update classification metrics (segmentation head)
            tp_seg += target * mask_cls
            tn_seg += (1 - target) * (1 - mask_cls)
            fp_seg += (1 - target) * mask_cls
            fn_seg += target * (1 - mask_cls)

            # Create output folder
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'masks', 'cls')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'masks', 'cls'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'masks', 'seg')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'masks', 'seg'))

            # Check if target and has_mask (i.e. neoplastic image with mask)
            if target and cls:

                # Update counter for localization
                loc_total_cls += 1

                # Check if biopsy in mask and update counters
                biopsy_soft = np.any(np.multiply(biopsy, mask_soft))
                loc_soft_cls += biopsy_soft
                tp_det_soft_cls += target * cls * biopsy_soft
                fn_det_soft_cls += target * cls * (1 - biopsy_soft)

                biopsy_plaus = np.any(np.multiply(biopsy, mask_plaus))
                loc_plaus_cls += biopsy_plaus
                tp_det_plaus_cls += target * cls * biopsy_plaus
                fn_det_plaus_cls += target * cls * (1 - biopsy_plaus)

                biopsy_sweet = np.any(np.multiply(biopsy, mask_sweet))
                loc_sweet_cls += biopsy_sweet
                tp_det_sweet_cls += target * cls * biopsy_sweet
                fn_det_sweet_cls += target * cls * (1 - biopsy_sweet)

                biopsy_hard = np.any(np.multiply(biopsy, mask_hard))
                loc_hard_cls += biopsy_hard
                tp_det_hard_cls += target * cls * biopsy_hard
                fn_det_hard_cls += target * cls * (1 - biopsy_hard)

                # Plot the results
                plt.subplots(1, 5)
                plt.subplot(1, 5, 1)
                plt.imshow(mask[0, :, :], cmap='gray')
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Prediction', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 2)
                plt.imshow(mask_soft, cmap='gray')
                for loc in biopsy_experts:
                    plt.scatter(loc[0], loc[1], c='green', s=1)
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Soft GT', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 3)
                plt.imshow(mask_plaus, cmap='gray')
                for loc in biopsy_experts:
                    plt.scatter(loc[0], loc[1], c='green', s=1)
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Plausible GT', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 4)
                plt.imshow(mask_sweet, cmap='gray')
                for loc in biopsy_experts:
                    plt.scatter(loc[0], loc[1], c='green', s=1)
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Sweet GT', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 5)
                plt.imshow(mask_hard, cmap='gray')
                for loc in biopsy_experts:
                    plt.scatter(loc[0], loc[1], c='green', s=1)
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Hard GT', fontsize=10)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_PATH, 'masks', 'cls', img_name + '.png'),
                            bbox_inches='tight')
                plt.close()

            else:
                fp_det_soft_cls += (1 - target) * cls
                tn_det_soft_cls += (1 - target) * (1 - cls)
                fn_det_soft_cls += target * (1 - cls)

                fp_det_plaus_cls += (1 - target) * cls
                tn_det_plaus_cls += (1 - target) * (1 - cls)
                fn_det_plaus_cls += target * (1 - cls)

                fp_det_sweet_cls += (1 - target) * cls
                tn_det_sweet_cls += (1 - target) * (1 - cls)
                fn_det_sweet_cls += target * (1 - cls)

                fp_det_hard_cls += (1 - target) * cls
                tn_det_hard_cls += (1 - target) * (1 - cls)
                fn_det_hard_cls += target * (1 - cls)

            # Check if target and has_mask (i.e. neoplastic image with mask)
            if target and mask_cls:

                # Update counter for localization
                loc_total_seg += 1

                # Check if biopsy in mask and update counters
                biopsy_soft = np.any(np.multiply(biopsy, mask_soft))
                loc_soft_seg += biopsy_soft
                tp_det_soft_seg += target * mask_cls * biopsy_soft
                fn_det_soft_seg += target * mask_cls * (1 - biopsy_soft)

                biopsy_plaus = np.any(np.multiply(biopsy, mask_plaus))
                loc_plaus_seg += biopsy_plaus
                tp_det_plaus_seg += target * mask_cls * biopsy_plaus
                fn_det_plaus_seg += target * mask_cls * (1 - biopsy_plaus)

                biopsy_sweet = np.any(np.multiply(biopsy, mask_sweet))
                loc_sweet_seg += biopsy_sweet
                tp_det_sweet_seg += target * mask_cls * biopsy_sweet
                fn_det_sweet_seg += target * mask_cls * (1 - biopsy_sweet)

                biopsy_hard = np.any(np.multiply(biopsy, mask_hard))
                loc_hard_seg += biopsy_hard
                tp_det_hard_seg += target * mask_cls * biopsy_hard
                fn_det_hard_seg += target * mask_cls * (1 - biopsy_hard)

                # Plot the results
                plt.subplots(1, 5)
                plt.subplot(1, 5, 1)
                plt.imshow(mask[0, :, :], cmap='gray')
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Prediction', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 2)
                plt.imshow(mask_soft, cmap='gray')
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Soft GT', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 3)
                plt.imshow(mask_plaus, cmap='gray')
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Plausible GT', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 4)
                plt.imshow(mask_sweet, cmap='gray')
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Sweet GT', fontsize=10)
                plt.axis('off')
                plt.subplot(1, 5, 5)
                plt.imshow(mask_hard, cmap='gray')
                plt.scatter(max_indices[0], max_indices[1], c='r', s=1)
                plt.title('Hard GT', fontsize=10)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_PATH, 'masks', 'seg', img_name + '.png'),
                            bbox_inches='tight')
                plt.close()

            else:
                fp_det_soft_seg += (1 - target) * mask_cls
                tn_det_soft_seg += (1 - target) * (1 - mask_cls)
                fn_det_soft_seg += target * (1 - mask_cls)

                fp_det_plaus_seg += (1 - target) * mask_cls
                tn_det_plaus_seg += (1 - target) * (1 - mask_cls)
                fn_det_plaus_seg += target * (1 - mask_cls)

                fp_det_sweet_seg += (1 - target) * mask_cls
                tn_det_sweet_seg += (1 - target) * (1 - mask_cls)
                fn_det_sweet_seg += target * (1 - mask_cls)

                fp_det_hard_seg += (1 - target) * mask_cls
                tn_det_hard_seg += (1 - target) * (1 - mask_cls)
                fn_det_hard_seg += target * (1 - mask_cls)

            # Add values to the dataframe
            cls_result = cls == target
            seg_result = mask_cls == target
            if target and cls:
                df.loc[logi] = [
                    img_name,
                    round(cls_pred.item(), 5),
                    round(mask_cls_logit.item(), 5),
                    cls_result,
                    seg_result,
                    biopsy_soft,
                    biopsy_plaus,
                    biopsy_sweet,
                    biopsy_hard
                ]
            elif target and mask_cls:
                df.loc[logi] = [
                    img_name,
                    round(cls_pred.item(), 5),
                    round(mask_cls_logit.item(), 5),
                    cls_result,
                    seg_result,
                    biopsy_soft,
                    biopsy_plaus,
                    biopsy_sweet,
                    biopsy_hard
                ]
            else:
                df.loc[logi] = [
                    img_name,
                    round(cls_pred.item(), 5),
                    round(mask_cls_logit.item(), 5),
                    cls_result,
                    seg_result,
                    'No TP',
                    'No TP',
                    'No TP',
                    'No TP'
                ]
            logi += 1

            # Process predicted mask and save to specified folder
            mask = mask.permute(1, 2, 0)
            maskpred = np.array(mask * 255, dtype=np.uint8)
            maskpred_pil = Image.fromarray(cv2.cvtColor(maskpred, cv2.COLOR_GRAY2RGB), mode='RGB')

            # Make folders
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'cls c-seg w')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'cls c-seg w'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'cls w-seg c')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'cls w-seg c'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'correct')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'correct'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'heatmap', 'wrong')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'heatmap', 'wrong'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'ROC')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'ROC'))

            #  Transform to heatmap
            heatmap = cv2.cvtColor(
                cv2.applyColorMap(maskpred, cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB,
            )
            heatmap = heatmap / 255.0

            # Define alpha value
            alphavalue = 0.5
            alpha = np.where(maskpred > 0.1, alphavalue, 0.0)

            # Process heatmap to PIL image, resize and convert to RGB
            heatmap = np.array(np.concatenate((heatmap, alpha), axis=-1) * 255, dtype=np.uint8)
            heatmap_pil = Image.fromarray(heatmap, mode='RGBA')
            w = int(image.size[0])
            h = int(image.size[1])
            heatmap_pil = heatmap_pil.resize(size=(w, h), resample=Image.NEAREST)
            heatmap_pil = heatmap_pil.convert('RGB')

            # Create original image with heatmap overlay
            composite = Image.blend(heatmap_pil, image, 0.6)
            draw = ImageDraw.Draw(composite)
            draw.text(
                (0, 0),
                "Cls: {:.3f}, Seg:{:.3f}".format(cls_pred.item(), mask_cls_logit.item()),
                (255, 255, 255),
            )

            # Save the composite image to specified folder
            if mask_cls != target and cls != target:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'wrong', img_name + '.jpg'))
            elif mask_cls != target and cls == target:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'cls c-seg w', img_name + '.jpg'))
            elif mask_cls == target and cls != target:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'cls w-seg c', img_name + '.jpg'))
            else:
                composite.save(os.path.join(OUTPUT_PATH, 'heatmap', 'correct', img_name + '.jpg'))

    # Compute accuracy, sensitivity and specificity for classification (classification head)
    accuracy_cls = (tp_cls + tn_cls) / (tp_cls + fn_cls + tn_cls + fp_cls)
    sensitivity_cls = tp_cls / (tp_cls + fn_cls)
    specificity_cls = tn_cls / (tn_cls + fp_cls + 1e-16)

    # Print accuracy, sensitivity and specificity
    print('\nClassification Performance (Classification head)')
    print('accuracy_cls: {:.4f}'.format(accuracy_cls))
    print('sensitivity_cls: {:.4f}'.format(sensitivity_cls))
    print('specificity_cls: {:.4f}'.format(specificity_cls + 1e-16))

    # Compute AUC for classification (classification head)
    auc = roc_auc_score(y_true, y_pred)
    print('auc_cls: {:.4f}'.format(auc))
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Write Classification performance to file
    if inf_set == 'Test':
        f_txt.write('### Test Set (Threshold CLS = {:.4f}, SEG = {:.4f}) ###'.format(thr_cls, thr_seg))

    f_txt.write('\nClassification Performance (Classification Head)')
    f_txt.write('\naccuracy_cls: {:.4f}'.format(accuracy_cls))
    f_txt.write('\nsensitivity_cls: {:.4f}'.format(sensitivity_cls))
    f_txt.write('\nspecificity_cls: {:.4f}'.format(specificity_cls))
    f_txt.write('\nauc_cls: {:.4f}\n'.format(auc))

    # Plot ROC curve for classification results and save to specified folder
    plt.plot(fpr, tpr, marker='.', label='Classification head')

    # Compute accuracy, sensitivity and specificity for classification (segmentation head)
    accuracy_seg = (tp_seg + tn_seg) / (tp_seg + fn_seg + tn_seg + fp_seg)
    sensitivity_seg = tp_seg / (tp_seg + fn_seg)
    specificity_seg = tn_seg / (tn_seg + fp_seg)

    # Print accuracy, sensitivity and specificity for segmentation
    print('\nClassification Performance (Segmentation Head)')
    print('accuracy_seg: {:.4f}'.format(accuracy_seg))
    print('sensitivity_seg: {:.4f}'.format(sensitivity_seg))
    print('specificity_seg: {:.4f}'.format(specificity_seg))

    # Compute AUC for segmentation
    auc = roc_auc_score(y_true, y_mask_pred)
    print('auc_seg: {:.4f}'.format(auc))
    fpr, tpr, thr = roc_curve(y_true, y_mask_pred)

    # Write Segmentation performance to file
    f_txt.write('\nClassification Performance (Segmentation Head)')
    f_txt.write('\naccuracy_seg: {:.4f}'.format(accuracy_seg))
    f_txt.write('\nsensitivity_seg: {:.4f}'.format(sensitivity_seg))
    f_txt.write('\nspecificity_seg: {:.4f}'.format(specificity_seg))
    f_txt.write('\nauc_seg: {:.4f}\n'.format(auc))

    # Calculate localization performance for classification (classification head)
    loc_soft_cls = loc_soft_cls / loc_total_cls
    loc_plaus_cls = loc_plaus_cls / loc_total_cls
    loc_sweet_cls = loc_sweet_cls / loc_total_cls
    loc_hard_cls = loc_hard_cls / loc_total_cls

    # Write localization performance to file
    f_txt.write('\nLocalization Performance (Classification Head)')
    f_txt.write('\nloc_soft_cls: {:.4f}'.format(loc_soft_cls))
    f_txt.write('\nloc_plaus_cls: {:.4f}'.format(loc_plaus_cls))
    f_txt.write('\nloc_sweet_cls: {:.4f}'.format(loc_sweet_cls))
    f_txt.write('\nloc_hard_cls: {:.4f}\n'.format(loc_hard_cls))

    # Calculate localization performance for classification (segmentation head)
    loc_soft_seg = loc_soft_seg / loc_total_seg
    loc_plaus_seg = loc_plaus_seg / loc_total_seg
    loc_sweet_seg = loc_sweet_seg / loc_total_seg
    loc_hard_seg = loc_hard_seg / loc_total_seg

    # Write localization performance to file
    f_txt.write('\nLocalization Performance (Segmentation Head)')
    f_txt.write('\nloc_soft_seg: {:.4f}'.format(loc_soft_seg))
    f_txt.write('\nloc_plaus_seg: {:.4f}'.format(loc_plaus_seg))
    f_txt.write('\nloc_sweet_seg: {:.4f}'.format(loc_sweet_seg))
    f_txt.write('\nloc_hard_seg: {:.4f}\n'.format(loc_hard_seg))

    # Calculate detection performance for classification (classification head)
    acc_det_soft_cls = (tp_det_soft_cls + tn_det_soft_cls) / (tp_det_soft_cls + fn_det_soft_cls + tn_det_soft_cls + fp_det_soft_cls)
    sens_det_soft_cls = tp_det_soft_cls / (tp_det_soft_cls + fn_det_soft_cls + 1e-16)
    spec_det_soft_cls = tn_det_soft_cls / (tn_det_soft_cls + fp_det_soft_cls + 1e-16)

    acc_det_plaus_cls = (tp_det_plaus_cls + tn_det_plaus_cls) / (tp_det_plaus_cls + fn_det_plaus_cls + tn_det_plaus_cls + fp_det_plaus_cls)
    sens_det_plaus_cls = tp_det_plaus_cls / (tp_det_plaus_cls + fn_det_plaus_cls + 1e-16)
    spec_det_plaus_cls = tn_det_plaus_cls / (tn_det_plaus_cls + fp_det_plaus_cls + 1e-16)

    acc_det_sweet_cls = (tp_det_sweet_cls + tn_det_sweet_cls) / (tp_det_sweet_cls + fn_det_sweet_cls + tn_det_sweet_cls + fp_det_sweet_cls)
    sens_det_sweet_cls = tp_det_sweet_cls / (tp_det_sweet_cls + fn_det_sweet_cls + 1e-16)
    spec_det_sweet_cls = tn_det_sweet_cls / (tn_det_sweet_cls + fp_det_sweet_cls + 1e-16)

    acc_det_hard_cls = (tp_det_hard_cls + tn_det_hard_cls) / (tp_det_hard_cls + fn_det_hard_cls + tn_det_hard_cls + fp_det_hard_cls)
    sens_det_hard_cls = tp_det_hard_cls / (tp_det_hard_cls + fn_det_hard_cls + 1e-16)
    spec_det_hard_cls = tn_det_hard_cls / (tn_det_hard_cls + fp_det_hard_cls + 1e-16)

    # Write detection performance to file
    f_txt.write('\nDetection Performance (Classification Head)')
    f_txt.write('\nacc_det_soft_cls: {:.4f}'.format(acc_det_soft_cls))
    f_txt.write('\nsens_det_soft_cls: {:.4f}'.format(sens_det_soft_cls))
    f_txt.write('\nspec_det_soft_cls: {:.4f}\n'.format(spec_det_soft_cls))

    f_txt.write('\nacc_det_plaus_cls: {:.4f}'.format(acc_det_plaus_cls))
    f_txt.write('\nsens_det_plaus_cls: {:.4f}'.format(sens_det_plaus_cls))
    f_txt.write('\nspec_det_plaus_cls: {:.4f}\n'.format(spec_det_plaus_cls))

    f_txt.write('\nacc_det_sweet_cls: {:.4f}'.format(acc_det_sweet_cls))
    f_txt.write('\nsens_det_sweet_cls: {:.4f}'.format(sens_det_sweet_cls))
    f_txt.write('\nspec_det_sweet_cls: {:.4f}\n'.format(spec_det_sweet_cls))

    f_txt.write('\nacc_det_hard_cls: {:.4f}'.format(acc_det_hard_cls))
    f_txt.write('\nsens_det_hard_cls: {:.4f}'.format(sens_det_hard_cls))
    f_txt.write('\nspec_det_hard_cls: {:.4f}\n'.format(spec_det_hard_cls))

    # Calculate detection performance for classification (segmentation head)
    acc_det_soft_seg = (tp_det_soft_seg + tn_det_soft_seg) / (tp_det_soft_seg + fn_det_soft_seg + tn_det_soft_seg + fp_det_soft_seg)
    sens_det_soft_seg = tp_det_soft_seg / (tp_det_soft_seg + fn_det_soft_seg + 1e-16)
    spec_det_soft_seg = tn_det_soft_seg / (tn_det_soft_seg + fp_det_soft_seg + 1e-16)

    acc_det_plaus_seg = (tp_det_plaus_seg + tn_det_plaus_seg) / (tp_det_plaus_seg + fn_det_plaus_seg + tn_det_plaus_seg + fp_det_plaus_seg)
    sens_det_plaus_seg = tp_det_plaus_seg / (tp_det_plaus_seg + fn_det_plaus_seg + 1e-16)
    spec_det_plaus_seg = tn_det_plaus_seg / (tn_det_plaus_seg + fp_det_plaus_seg + 1e-16)

    acc_det_sweet_seg = (tp_det_sweet_seg + tn_det_sweet_seg) / (tp_det_sweet_seg + fn_det_sweet_seg + tn_det_sweet_seg + fp_det_sweet_seg)
    sens_det_sweet_seg = tp_det_sweet_seg / (tp_det_sweet_seg + fn_det_sweet_seg + 1e-16)
    spec_det_sweet_seg = tn_det_sweet_seg / (tn_det_sweet_seg + fp_det_sweet_seg + 1e-16)

    acc_det_hard_seg = (tp_det_hard_seg + tn_det_hard_seg) / (tp_det_hard_seg + fn_det_hard_seg + tn_det_hard_seg + fp_det_hard_seg)
    sens_det_hard_seg = tp_det_hard_seg / (tp_det_hard_seg + fn_det_hard_seg + 1e-16)
    spec_det_hard_seg = tn_det_hard_seg / (tn_det_hard_seg + fp_det_hard_seg + 1e-16)

    # Write detection performance to file
    f_txt.write('\nDetection Performance (Segmentation Head)')
    f_txt.write('\nacc_det_soft_seg: {:.4f}'.format(acc_det_soft_seg))
    f_txt.write('\nsens_det_soft_seg: {:.4f}'.format(sens_det_soft_seg))
    f_txt.write('\nspec_det_soft_seg: {:.4f}\n'.format(spec_det_soft_seg))

    f_txt.write('\nacc_det_plaus_seg: {:.4f}'.format(acc_det_plaus_seg))
    f_txt.write('\nsens_det_plaus_seg: {:.4f}'.format(sens_det_plaus_seg))
    f_txt.write('\nspec_det_plaus_seg: {:.4f}\n'.format(spec_det_plaus_seg))

    f_txt.write('\nacc_det_sweet_seg: {:.4f}'.format(acc_det_sweet_seg))
    f_txt.write('\nsens_det_sweet_seg: {:.4f}'.format(sens_det_sweet_seg))
    f_txt.write('\nspec_det_sweet_seg: {:.4f}\n'.format(spec_det_sweet_seg))

    f_txt.write('\nacc_det_hard_seg: {:.4f}'.format(acc_det_hard_seg))
    f_txt.write('\nsens_det_hard_seg: {:.4f}'.format(sens_det_hard_seg))
    f_txt.write('\nspec_det_hard_seg: {:.4f}\n'.format(spec_det_hard_seg))

    # Plot ROC curve for segmentation results and save to specified folder
    plt.plot(fpr, tpr, marker='.', label='Max segmentation Value')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    major_ticks = np.arange(0.0, 1.01, 0.05)
    plt.xticks(major_ticks, fontsize='x-small')
    plt.yticks(major_ticks)
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.grid(True)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.title('ROC AUC')
    plt.savefig(os.path.join(OUTPUT_PATH, 'ROC', 'auc_curve.jpg'))
    plt.close()

    # Save dataframe as csv file
    df.to_excel(os.path.join(OUTPUT_PATH, 'cls_scores.xlsx'))


"""""" """""" """"""
"""" EXECUTION """
"""""" """""" """"""


if __name__ == '__main__':
    """SPECIFY PATH FOR SAVING"""
    SAVE_DIR = os.path.join(os.getcwd(), "experiments")

    """ARGUMENT PARSER"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experimentnames', type=list_of_settings)
    parser.add_argument('--evaluate_sets', type=list_of_settings)
    parser.add_argument('--textfile', type=str, default='Results.txt')
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--sens_val', type=float, default=0.9)
    inference_opt = parser.parse_args()

    """LOOP OVER ALL EXPERIMENTS"""
    for exp_name in inference_opt.experimentnames:
        # EXTRACT INFORMATION FROM PARAMETERS USED IN EXPERIMENT
        f = open(os.path.join(SAVE_DIR, exp_name, 'params.json'))
        data = json.load(f)
        opt = {
            'experimentname': exp_name,
            'backbone': data['backbone'],
            'seg_branch': data['seg_branch'],
            'imagesize': data['imagesize'],
            'num_classes': data['num_classes'],
            'label_smoothing': data['label_smoothing'],
            'evaluate_sets': inference_opt.evaluate_sets,
            'radius': inference_opt.radius,
            'sens_val': inference_opt.sens_val,
            'textfile': inference_opt.textfile,
        }
        opt = argparse.Namespace(**opt)

        # Create text file for writing results
        f = open(os.path.join(SAVE_DIR, exp_name, opt.textfile), 'x')
        f_txt = open(os.path.join(SAVE_DIR, exp_name, opt.textfile), 'a')

        # Loop over all sets
        for inf_set in opt.evaluate_sets:
            if inf_set == 'Test':
                CACHE_PATH = os.path.join(os.getcwd(), "cache folders", "cache")
                OUTPUT_PATH = os.path.join(SAVE_DIR, exp_name, 'Inference', 'Test Set')
            else:
                raise ValueError

            # Run inference
            thr_cls, thr_seg = run_val(opt=opt, exp_name=exp_name)
            run(opt=opt, f_txt=f_txt, exp_name=exp_name, inf_set=inf_set, thr_cls=thr_cls, thr_seg=thr_seg)

        # Close text file
        f_txt.close()
