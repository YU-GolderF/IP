import sys
from pathlib import Path

# Mirror the sys.path setup used by app.py so tests can import the same packages.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # New folder/
for _pkg in ("algorithms", "core", "reporting"):
    _path = str(_PROJECT_ROOT / _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
