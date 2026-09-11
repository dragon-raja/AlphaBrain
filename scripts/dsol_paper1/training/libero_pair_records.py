"""Compatibility alias for the dataset record codec."""
import sys
from AlphaBrain.common import pair_records
sys.modules[__name__] = pair_records
