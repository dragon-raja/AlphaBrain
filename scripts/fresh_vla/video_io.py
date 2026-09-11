"""Compatibility import for historical CABI callers."""
import importlib
import sys
sys.modules[__name__]=importlib.import_module('scripts.vla_shared.historical_video_io')
