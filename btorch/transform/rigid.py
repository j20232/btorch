import torch
from btorch.transform.base import batch_identity, homogeneous_bottom


def rigid_transform(locations, poses, scales, dtype=torch.float32,
                    device="cuda" if torch.cuda.is_available() else "cpu"):
    loc_tensor = locations2tensor(locations)
    rot_tensor = rotations2tensor(poses)
    scale_tensor = scales2tensor(scales)
    return loc_tensor @ rot_tensor @ scale_tensor


def locations2tensor(locations, dtype=torch.float32,
                     device="cuda" if torch.cuda.is_available() else "cpu"):
    if len(locations.shape) == 1:
        locations = locations.view(1, -1)
    batch_num = locations.shape[0]
    eyes = batch_identity(batch_num, size=3, device=device)
    bottom = homogeneous_bottom(batch_num, device=device).view(-1, 1, 4)
    top = torch.cat([eyes, locations.unsqueeze(2)], axis=2)
    return torch.cat([top, bottom], axis=1)


def rotations2tensor(poses, dtype=torch.float32,
                     device="cuda" if torch.cuda.is_available() else "cpu"):
    # `poses` should be represented as normalized axis angles (same as SMPL context)
    if len(poses.shape) == 1:
        poses = poses.view(1, 3)
    if len(poses.shape) == 2:
        poses = poses.view(-1, 1, 3)

    batch_num = poses.shape[0]
    rotation = rodrigues(poses)
    right = torch.zeros((batch_num, 3), dtype=dtype).view(-1, 3, 1).to(device)
    top = torch.cat([rotation, right], axis=2)
    bottom = homogeneous_bottom(batch_num, device=device).view(-1, 1, 4)
    return torch.cat([top, bottom], axis=1)


def scales2tensor(scales, dtype=torch.float32,
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    if len(scales.shape) == 1:
        scales = scales.view(1, 3)
    batch_num = scales.shape[0]
    scale = torch.diag_embed(scales, offset=0, dim1=-2, dim2=-1)
    right = torch.zeros((batch_num, 3), dtype=dtype).view(-1, 3, 1).to(device)
    top = torch.cat([scale, right], axis=2)
    bottom = homogeneous_bottom(batch_num, device=device).view(-1, 1, 4)
    return torch.cat([top, bottom], axis=1)


def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size, 1, 3].
    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].
    """

    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    z_stick = torch.zeros(theta_dim, dtype=r.dtype).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=r.dtype).unsqueeze(dim=0)
              + torch.zeros((theta_dim, 3, 3), dtype=r.dtype)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    cos = torch.cos(theta)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R
