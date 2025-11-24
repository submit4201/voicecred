"""
conftest.py

Make the project's src/ importable for tests and (optionally) add a local venv's
site-packages to sys.path if it exists. This avoids forcing a process exec or
mutating PYTHONHOME/PYTHONPATH in a brittle way.
"""

from pathlib import Path
import sys
import site
import os

# Project root is the parent of this tests directory
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Optional: path to a local venv. Can be overridden by environment var VC_VENV.
# Default kept for compatibility with your original layout.
DEFAULT_VENV = Path(r"C:\workspace\voicecred\.conda")
VENV = Path(os.environ.get("VC_VENV", str(DEFAULT_VENV)))

def _venv_site_packages(venv_path: Path):
    """Return the site-packages path for a venv if it exists, else None."""
    if not venv_path.exists():
        return None

    # Windows venv layout
    win_sp = venv_path / "Lib" / "site-packages"
    if win_sp.exists():
        return str(win_sp)

    # POSIX venv layout: lib/pythonX.Y/site-packages
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    posix_sp = venv_path / "lib" / py_ver / "site-packages"
    if posix_sp.exists():
        return str(posix_sp)

    return None

# Add src first so tests import project packages before installed ones
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# If a venv site-packages is present, add it (after src) so tests can find deps
venv_sp = _venv_site_packages(VENV)
if venv_sp and venv_sp not in sys.path:
    sys.path.insert(1, venv_sp)
    # Also register with site for pkgutil/stdlib friendliness
    site.addsitedir(venv_sp)

# Helpful debug info when running pytest with -q/--capture=tee-sys suppressed:
if os.environ.get("PYTEST_ADD_SYS_PATH_DEBUG"):
    print("conftest: ROOT =", ROOT)
    print("conftest: SRC =", SRC)
    print("conftest: VENV =", VENV)
    print("conftest: venv site-packages =", venv_sp)
    print("conftest: sys.path[0:5] =", sys.path[:5])
