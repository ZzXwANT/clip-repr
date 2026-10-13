import torch
from config import config

def create_optimizer(model):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_name = name.split('.')[-1]

        if param_name in ('bias', 'gamma', 'beta', 'logit_scale'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=config.learning_rate,
    )
    
    return optimizer