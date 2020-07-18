
from btorch.utils import seed

__version__ = '0.1.0'

if True:
    import importlib
    importlib.reload(seed)
    from btorch.utils.seed import seed_everything
