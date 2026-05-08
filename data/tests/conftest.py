import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))      # data/   (for prepare_sft)
sys.path.insert(0, str(_HERE.parents[2]))      # repo root (for scripts.*)
