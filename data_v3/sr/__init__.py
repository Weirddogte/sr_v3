import sys as _sys
from pathlib import Path as _Path

# Ensure the data_v3/ root is on sys.path so `import sr` works in spawned
# DataLoader worker processes, which start fresh and don't inherit the
# parent's runtime sys.path modifications.
_pkg_root = str(_Path(__file__).parent.parent)
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)
