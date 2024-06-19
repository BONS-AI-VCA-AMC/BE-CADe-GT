"""IMPORT PACKAGES"""
import torch

"""""" """""" """""" """""" """""" """""" """"""
"""" DEFINE HELPER FUNCTIONS FOR OPTIMIZER"""
"""""" """""" """""" """""" """""" """""" """"""


def construct_optimizer(optim, parameters, lr):
    # Define possible choices
    if optim == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-07,
            amsgrad=True,
            weight_decay=1e-4,
        )
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise Exception('Unexpected Optimizer {}'.format(optim))

    return optimizer


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
"""" DEFINE HELPER FUNCTIONS FOR LEARNING RATE SCHEDULER"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


def construct_scheduler(schedule, optimizer, lr, metric="val_loss_combine"):
    # Define possible choices
    if schedule == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            min_lr=lr / 1000,
        )

        return {"scheduler": scheduler, "monitor": metric, "interval": "epoch"}

    elif schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

        return {"scheduler": scheduler, "interval": "epoch"}

    elif schedule == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=2,
            T_mult=2,
            eta_min=lr / 1000,
            last_epoch=-1,
        )

        return {"scheduler": scheduler, "interval": "epoch"}

    else:
        return None
