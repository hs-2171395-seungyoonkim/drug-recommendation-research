"""
Thin CLI wrapper so `safedrug.train.main()` can actually be run from the repo
root. There's no pyproject.toml/setup.py in this project, so `src/` never
reaches sys.path outside of pytest's conftest.py - this script does the same
sys.path insertion conftest.py does, then delegates to safedrug.train.main().

Run from the ServerityMed repo root:
  C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts/run_safedrug_train.py --organ_function
  (omit the flag for baseline)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safedrug.train import main

if __name__ == "__main__":
    main()
