import torch
import torch.nn.functional as F
from networks import DQN_CNN, DQN_LSTM


def model_factory(model_type):
    if model_type == "CNN":
        return DQN_CNN
    elif model_type == "LSTM":
        return DQN_LSTM
    else:
        raise ValueError("Unknown loss: {}".format(model_type))


def loss_function_factory(loss_type):
    if loss_type == "l2":
        return F.mse_loss
    elif loss_type == "l1":
        return F.l1_loss
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss
    else:
        raise ValueError("Unknown loss: {}".format(loss_type))


def optimizer_factory(optimizer_type):
    if optimizer_type == "adam":
        return torch.optim.Adam
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop
    elif optimizer_type == "adamw":
        return torch.optim.AdamW
    elif optimizer_type == "sgd":
        return torch.optim.SGD
    elif optimizer_type == "sparseadam":
        return torch.optim.SparseAdam
    elif optimizer_type == "asgd":
        return torch.optim.ASGD
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_type))
