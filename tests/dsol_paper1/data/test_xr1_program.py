"""The three partitions must reproduce the original97, with no extra work."""
import pytest

from AlphaBrain.research.dsol.data.xr1_program import phase_views


def test_partition_union_and_order():
    all_views = ['canonical', *[f'view_{i}' for i in range(96)]]
    small = ['canonical', 'view_2', 'view_31', 'view_30', 'view_11']
    partitions = [phase_views({'all_views': all_views}, {'panel_views': small}, p)
                  for p in ('canonical', 'perturbed', 'remaining')]
    assert [len(p) for p in partitions] == [1, 4, 92]
    assert len(set(sum(partitions, []))) == 97
    assert set(sum(partitions, [])) == set(all_views)
    assert partitions[-1] == [v for v in all_views if v not in small]
    assert sum(40 * len(p) * 4 for p in partitions) == 15520


def test_bad_full_catalog_rejected():
    with pytest.raises(ValueError, match='97-view'):
        phase_views({'all_views': ['canonical'] * 97},
                    {'panel_views': ['canonical', 'a', 'b', 'c', 'd']}, 'remaining')


def test_no_full_catalog_needed_for_legacy_small_panel():
    assert phase_views({}, {'panel_views': ['canonical', 'a', 'b', 'c', 'd']}, 'perturbed') == ['a', 'b', 'c', 'd']
