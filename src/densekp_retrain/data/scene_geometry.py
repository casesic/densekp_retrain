"""Legacy import shim; the shared generator implementation lives in human_data_generator."""
from pathlib import Path
import sys

_new_src = Path(__file__).resolve().parents[4] / 'human_data_generator' / 'src'
if not _new_src.is_dir():
    raise ImportError(f'human_data_generator sibling project missing: {_new_src}')
if str(_new_src) not in sys.path:
    sys.path.insert(0, str(_new_src))

from human_data_generator.data.scene_geometry import person_geometry, transform

__all__ = ['person_geometry', 'transform']
