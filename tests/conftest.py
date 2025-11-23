import os
import sys

# Ensure src is on python path for tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
VENV= "C:\\workspace\\voicecred\\.conda"
# add venv site-packages to path
SITE_PACKAGES = os.path.join(VENV, 'Lib', 'site-packages')
# ensure using correct python from venv
if not sys.executable.startswith(VENV):
    os.environ["PYTHONHOME"] = VENV
    os.environ["PYTHONPATH"] = SITE_PACKAGES
# run test with .conda python at VENV
if sys.executable != os.path.join(VENV, 'python.exe'):
    os.execv(os.path.join(VENV, 'python.exe'), [os.path.join(VENV, 'python.exe')] + sys.argv)
if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
