"""Historical import compatibility; active callers use the public module."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from densekp_retrain.rendering.blender.groom import add_groom
