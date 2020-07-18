import bpy
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
    source_vertices = bnp.obj2np(source, as_homogeneous=True)
    source_location = bnp.location2np(source)
    print("Source vertices: ", source_vertices.shape)
    print("Source location", source_location)

    # Target
    target = bpy.context.scene.objects["Target"]
    bnp.change_rotation_mode(target, "AXIS_ANGLE")
    target_vertices = bnp.obj2np(target, as_homogeneous=True)
    target_location = bnp.location2np(target)
    print("Target vertices: ", target_vertices.shape)
    print("Target location", target_location)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_trans = torch.tensor(source_location, requires_grad=True).to(device)
    print("Initial translation", init_trans)
