"""Compatibility alias; image primitives belong to the core package."""
import sys
from AlphaBrain.common import images
sys.modules[__name__] = images
