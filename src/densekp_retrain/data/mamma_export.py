"""Legacy import shim; new MAMMA observation export is owned by human_data_generator."""
from pathlib import Path
import sys

_new_src = Path(__file__).resolve().parents[4] / 'human_data_generator' / 'src'
if not _new_src.is_dir():
    raise ImportError(f'human_data_generator sibling project missing: {_new_src}')
if str(_new_src) not in sys.path:
    sys.path.insert(0, str(_new_src))

from human_data_generator.data.mamma_export import export_existing

__all__ = ['export_existing']
