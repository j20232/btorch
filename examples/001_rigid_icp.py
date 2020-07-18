import bpy
import numpy as np
import torch
import sys
from pathlib import Path
import importlib

# Please change LIBRARY_ROOT_PATH according to your environment
LIBRARY_ROOT_PATH = Path(".").resolve().parent
sys.path.append(str(LIBRARY_ROOT_PATH))

BNP_PATH = LIBRARY_ROOT_PATH / "third_party" / "bnp"
sys.path.append(str(BNP_PATH))

if __name__ == '__main__':
    import bnp
    import btorch
    importlib.reload(bnp)
    importlib.reload(btorch)
    btorch.seed_everything()

    # Source
    source = bpy.context.scene.objects["Source"]
    bnp.change_rotation_mode(source, "AXIS_ANGLE")

    # Target
    target = bpy.context.scene.objects["Target"]
    bnp.change_rotation_mode(target, "AXIS_ANGLE")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # translation
    trans = bnp.location2np(source)
    init_trans = torch.tensor(trans).view(1, -1).to(device).detach().requires_grad_(True)
    print("Initial translation: ", init_trans)

    # rotation
    pose = bnp.rotation2np(source)
    pose = bnp.normalize_axis_angle(pose)
    pose = pose[:, 0] * pose[:, 1:4]
    init_pose = torch.tensor(pose).view(1, -1).to(device).detach().requires_grad_(True)
    print("Initial rotation: ", init_trans)

    # scale
    scale = bnp.scale2np(source)
    init_scale = torch.tensor(scale).view(1, -1).to(device).detach().requires_grad_(True)
    print("Initial scale: ", init_scale)

    verts = bnp.obj2np(source, is_local=True)

    source_vertices = torch.tensor(bnp.obj2np(source, is_local=True, as_homogeneous=True).astype(np.float32)).view(-1, 1, 4).to(device)
    target_vertices = torch.tensor(bnp.obj2np(target, as_homogeneous=True)).view(-1, 1, 4).to(device)
    print("Source vertices: ", source_vertices.shape)
    print("Target vertices: ", target_vertices.shape)

    epochs = 600
    lr = 0.1
    verbose = 100
    esr = 100
    current = 0

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([init_trans, init_pose, init_scale], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    best_loss = 10000
    best_trans = np.zeros([0, 0, 0])
    for epoch in range(epochs):
        optimizer.zero_grad()
        rigid_trans = btorch.rigid_transform(init_trans, init_pose, init_scale).transpose(1, 2)
        loss = loss_fn(source_vertices @ rigid_trans, target_vertices)

        if loss < best_loss:
            best_loss = loss
            best_trans = init_trans.detach().cpu().numpy()[0]
            best_pose = init_pose.detach().cpu().numpy()[0]
            best_scale = init_scale.detach().cpu().numpy()[0]
            current = 0
        else:
            current += 1

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % verbose == 0:
            print("Epoch ", epoch, "Loss: ", loss.detach().cpu().numpy(), "trans: ", best_trans, "pose: ", best_pose, "scale: ", best_scale)
        if current > esr:
            print("Early stopping at ", epoch)
            break

        del rigid_trans
        torch.cuda.empty_cache()

    optimized_source = bpy.context.scene.objects["OptimizedSource"]
    optimized_source.location = (best_trans[0], best_trans[1], best_trans[2])
    norm = np.sqrt(best_pose[0] ** 2 + best_pose[1] ** 2 + best_pose[2] ** 2)
    optimized_source.rotation_axis_angle = (norm, best_pose[0], best_pose[1], best_pose[2])
    optimized_source.scale = (best_scale[0], best_scale[1], best_scale[2])
