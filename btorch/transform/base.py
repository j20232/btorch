import torch


def batch_identity(batch_num, size=4, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"):
    x = torch.eye(size, dtype=dtype).to(device)
    x = x.reshape((1, size, size))
    return x.repeat(batch_num, 1, 1)


def homogeneous_bottom(batch_num, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"):
    zero = torch.zeros((batch_num, 3), dtype=dtype).to(device)
    one = torch.ones((batch_num, 1), dtype=dtype).to(device)
    return torch.cat([zero, one], axis=1)
