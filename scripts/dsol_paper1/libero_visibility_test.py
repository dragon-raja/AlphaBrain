from types import SimpleNamespace

import pytest

from scripts.dsol_paper1.libero_visibility import _entity_geom_ids


def fake_env(*geom_names: str) -> SimpleNamespace:
    model = SimpleNamespace(geom_names=geom_names)
    sim = SimpleNamespace(model=model)
    return SimpleNamespace(env=SimpleNamespace(sim=sim))


def test_entity_geom_ids_matches_object_instance() -> None:
    ids, source = _entity_geom_ids(
        fake_env("cream_cheese_1_g0", "cream_cheese_1_g1", "bowl_1_g0"),
        "cream_cheese_1",
    )
    assert ids.tolist() == [0, 1]
    assert source == "cream_cheese_1"


def test_entity_geom_ids_resolves_semantic_region_to_fixture() -> None:
    ids, source = _entity_geom_ids(
        fake_env("wooden_cabinet_1_g0", "wooden_cabinet_1_g1"),
        "wooden_cabinet_1_top_region",
    )
    assert ids.tolist() == [0, 1]
    assert source == "wooden_cabinet_1"


def test_entity_geom_ids_fails_closed_for_unknown_region() -> None:
    with pytest.raises(KeyError, match="no MuJoCo geoms"):
        _entity_geom_ids(fake_env("plate_1_g0"), "missing_fixture_1_top_region")
