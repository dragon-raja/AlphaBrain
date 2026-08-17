from __future__ import annotations

import json
from pathlib import Path
import unittest


SCHEMA_ROOT = Path(__file__).parents[2] / "schemas/dsol_paper1"
ED_GATES = {
    "current_full_observation_ambiguous",
    "latent_decision_sets_incompatible",
    "no_shared_safe_recovery_action",
    "candidate_snapshot_disambiguates",
    "matched_uninformative_control_exists",
}


def load(name: str) -> dict[str, object]:
    return json.loads((SCHEMA_ROOT / name).read_text())


def local_refs(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref" and isinstance(child, str) and not child.startswith("#"):
                yield child.split("#", 1)[0]
            yield from local_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from local_refs(child)


class SchemaContractTests(unittest.TestCase):
    def test_all_schemas_parse_and_local_refs_exist(self) -> None:
        schemas = sorted(SCHEMA_ROOT.glob("*.schema.json"))
        self.assertEqual(len(schemas), 8)
        for path in schemas:
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text())
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                if path.name != "common.schema.json":
                    self.assertIs(schema.get("additionalProperties"), False)
                for ref in local_refs(schema):
                    self.assertTrue((SCHEMA_ROOT / ref).is_file(), ref)

    def test_ed_schema_freezes_exact_five_gates(self) -> None:
        relation = load("relation_case.schema.json")
        gate_schema = relation["$defs"]["e_d_gates"]  # type: ignore[index]
        required = set(gate_schema["required"]) - {"status"}  # type: ignore[index]
        properties = set(gate_schema["properties"]) - {"status"}  # type: ignore[index]
        self.assertEqual(required, ED_GATES)
        self.assertEqual(properties, ED_GATES)
        self.assertIs(gate_schema["additionalProperties"], False)  # type: ignore[index]

    def test_temporal_schema_keeps_all_strata(self) -> None:
        temporal = load("temporal_gate.schema.json")
        classification = temporal["$defs"]["classification"]  # type: ignore[index]
        strata = set(classification["properties"]["temporal_class"]["enum"])  # type: ignore[index]
        self.assertEqual(strata, {"S0", "S1", "S2", None})

    def test_release_schema_is_fail_closed(self) -> None:
        release = load("release_receipt.schema.json")
        gates = release["properties"]["gates"]  # type: ignore[index]
        self.assertEqual(set(gates["required"]), {f"B{index}" for index in range(1, 8)})  # type: ignore[index]
        self.assertEqual(set(release["properties"]["decision"]["enum"]), {"HOLD", "RELEASE"})  # type: ignore[index]
        self.assertTrue(release["allOf"])


if __name__ == "__main__":
    unittest.main()
