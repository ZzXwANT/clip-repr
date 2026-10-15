import torch

def create_scheduler(optimizer, total_steps, warmup_steps):
    if not 0 <= warmup_steps < total_steps: raise ValueError("需要满足 0 <= warmup_steps < total_steps，warmup 必须少于剩余训练步数")
    # warmup + 余弦衰减lr
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0.0,
    )
    scheduler = cosine_scheduler
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    return scheduler


def restore_scheduler(optimizer, state_dict):
    if "_schedulers" in state_dict:
        warmup_steps = state_dict["_milestones"][0]
        total_steps = warmup_steps + state_dict["_schedulers"][-1]["T_max"]
    else:
        warmup_steps = 0
        total_steps = state_dict["T_max"]
    scheduler = create_scheduler(optimizer, total_steps, warmup_steps)
    scheduler.load_state_dict(state_dict)
    return scheduler
