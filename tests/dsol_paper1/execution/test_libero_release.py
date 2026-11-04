from pathlib import Path
from tests.dsol_paper1.paths import ROOT
from AlphaBrain.research.dsol.execution.source_release import libero_runtime_sources


def test_libero_release_contains_lazy_simulator_dependencies():
    root=ROOT
    sources=set(libero_runtime_sources(root))
    for name in ('audit_libero_hdf5_restore','evaluate_dsol_libero_hdf5_views',
                 'standard_initialization_v1','render_bridge_common_v1','scan_libero_hdf5_views'):
        assert Path('scripts/dsol_paper1/runtime/'+name+'.py') in sources
    assert Path('benchmarks/LIBERO/search_simulator.py') in sources
