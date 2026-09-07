"""Repo-root and data paths, computed once.

`ROOT` is the repo root (the parent of this package). Every module and notebook
reaches `data/` through `DATA` here instead of hand-rolling `Path(__file__)`
walks, so moving a file between subpackages no longer breaks its data paths.

`CARRYIA_ROOT` overrides the root — the one seam the Docker image needs, where
the package installs to a different prefix than the mounted data dir.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CARRYIA_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
