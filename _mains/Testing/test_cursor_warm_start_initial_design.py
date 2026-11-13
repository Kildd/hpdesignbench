"""Loader for the ranked warm-start InitialDesign."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.ioh_core.algorithms.warm_start import (
    initial_design_prefix,
    load_initial_design,
    require_warm_start_budget,
    validate_prefix_bounds,
    validate_warm_start_candidate,
)

_VECTOR_A = [0, 0, 0, 0, 0, 0, 0, 0, 0]
_VECTOR_B = [1, 0, 0, 0, 0, 0, 0, 0, 1]


def _payload(
    problem_id: str = "L8100_B2",
    vectors: list | object | None = None,
    **overrides,
):
    chosen = [_VECTOR_A, _VECTOR_B] if vectors is None else vectors
    body = {
        "schema_version": 1,
        "problem_id": problem_id,
        "n": len(chosen) if isinstance(chosen, list) else 1,
        "vectors": chosen,
    }
    body.update(overrides)
    if "n" not in overrides and isinstance(body["vectors"], list):
        body["n"] = len(body["vectors"])
    return body


def _write(directory: Path, payload: dict, name: str = "initial_design.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadInitialDesign(unittest.TestCase):
    def test_schema_1_loads_and_schema_2_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            current = _write(Path(tmp), _payload())
            loaded = load_initial_design(
                current,
                algorithm_name="SCIPY_COBYLA_WARM_START",
                problem_id="L8100_B2",
            )
            old = _write(Path(tmp), _payload(schema_version=2), name="old.json")
            with self.assertRaises(ValueError) as raised:
                load_initial_design(
                    old,
                    algorithm_name="SCIPY_COBYLA_WARM_START",
                    problem_id="L8100_B2",
                )
        self.assertEqual(loaded, (tuple(_VECTOR_A), tuple(_VECTOR_B)))
        self.assertIn("schema_version", str(raised.exception))
        self.assertIn("1", str(raised.exception))

    def test_prefix_of_one_and_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), _payload())
            one = initial_design_prefix(
                path, 1, algorithm_name="SCIPY_COBYLA_WARM_START", problem_id="L8100_B2"
            )
            two = initial_design_prefix(
                path, 2, algorithm_name="SCIPY_COBYLA_WARM_START", problem_id="L8100_B2"
            )
        self.assertEqual(one, (tuple(_VECTOR_A),))
        self.assertEqual(two, (tuple(_VECTOR_A), tuple(_VECTOR_B)))

    def test_short_file_raises_before_any_vector_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), _payload())
            with self.assertRaises(ValueError) as raised:
                initial_design_prefix(
                    path,
                    3,
                    algorithm_name="SCIPY_COBYLA_WARM_START",
                    problem_id="L8100_B2",
                )
        message = str(raised.exception)
        self.assertIn("needs 3 initial designs", message)
        self.assertIn("contains 2", message)

    def test_missing_path_names_the_predesign_grid(self) -> None:
        missing = Path(tempfile.gettempdir()) / "no-such-initial-design.json"
        with self.assertRaises(ValueError) as raised:
            load_initial_design(
                missing,
                algorithm_name="SCIPY_COBYLA_WARM_START",
                problem_id="L8100_B2",
            )
        message = str(raised.exception)
        self.assertIn("has no InitialDesign", message)
        self.assertIn("running the predesign grid", message)

    def test_missing_shelf_names_predesign_and_not_runs(self) -> None:
        from core.ioh_core.workflows.predesign_paths import (
            predesign_initial_design_path,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = predesign_initial_design_path(tmp, "L8100_B2")
            with self.assertRaises(ValueError) as raised:
                load_initial_design(
                    path,
                    algorithm_name="SCIPY_COBYLA_WARM_START",
                    problem_id="L8100_B2",
                )
        message = str(raised.exception)
        self.assertIn(str(path), message)
        self.assertIn("predesign", message)
        self.assertNotIn("runs/predesign", message.replace("\\", "/"))
        self.assertNotIn("runs\\predesign", message)

    def test_wrong_problem_id_returns_no_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), _payload(problem_id="L9450_B2", vectors=[_VECTOR_A]))
            with self.assertRaises(ValueError) as raised:
                load_initial_design(
                    path,
                    algorithm_name="SCIPY_COBYLA_WARM_START",
                    problem_id="L8100_B2",
                )
        message = str(raised.exception)
        self.assertIn(str(path), message)
        self.assertIn("L9450_B2", message)
        self.assertIn("L8100_B2", message)

    def test_n_disagrees_with_vector_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp),
                _payload(vectors=[_VECTOR_A], n=3),
            )
            with self.assertRaises(ValueError) as raised:
                load_initial_design(
                    path,
                    algorithm_name="SCIPY_COBYLA_WARM_START",
                    problem_id="L8100_B2",
                )
        message = str(raised.exception)
        self.assertIn("n=3", message)
        self.assertIn("1", message)

    def test_json_true_is_rejected(self) -> None:
        cases = (
            {"schema_version": True},
            {"n": True, "vectors": [_VECTOR_A]},
            {"vectors": [[True, 0, 0, 0, 0, 0, 0, 0, 0]]},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as tmp:
                    path = _write(Path(tmp), _payload(**overrides))
                    with self.assertRaises(ValueError):
                        load_initial_design(
                            path,
                            algorithm_name="SCIPY_COBYLA_WARM_START",
                            problem_id="L8100_B2",
                        )

    def test_vectors_must_be_a_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), _payload(vectors={"0": _VECTOR_A}, n=1))
            with self.assertRaises(ValueError) as raised:
                load_initial_design(
                    path,
                    algorithm_name="SCIPY_COBYLA_WARM_START",
                    problem_id="L8100_B2",
                )
        self.assertIn("not a list", str(raised.exception))

    def test_bool_prefix_length_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), _payload(vectors=[_VECTOR_A]))
            with self.assertRaises(ValueError):
                initial_design_prefix(
                    path,
                    True,  # type: ignore[arg-type]
                    algorithm_name="SCIPY_COBYLA_WARM_START",
                    problem_id="L8100_B2",
                )

    def test_short_budget(self) -> None:
        with self.assertRaises(ValueError) as raised:
            require_warm_start_budget("X", 5, 4)
        self.assertIn(
            "warm start for X needs 5 initial evaluations but max_evaluations is 4.",
            str(raised.exception),
        )

    def test_invalid_second_vector_does_not_call_the_problem(self) -> None:
        calls: list[object] = []

        def problem(candidate: object) -> float:
            calls.append(candidate)
            return 0.0

        valid = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        invalid = (0, 0, 0, 0, 0, 0, 0, 0, 9)
        with self.assertRaises(Exception):
            validate_prefix_bounds(
                (valid, invalid),
                lower_bounds=(0,) * 9,
                upper_bounds=(1,) * 9,
            )
            problem(valid)
        self.assertEqual(calls, [])

    def test_bounds_check_accepts_a_problem_id(self) -> None:
        class _Context:
            problem_id = "L10125_A1"

            def decode(self, _candidate: object) -> dict[str, float]:
                return {"geom_span_mm": 10125.0}

        class _Problem:
            hdb_eval_context = _Context()

        vector = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        got = validate_warm_start_candidate(
            _Problem(),
            vector,
            lower_bounds=(0,) * 9,
            upper_bounds=(1,) * 9,
            algorithm_name="PYNOMAD_MADS_PB_WARM_START",
        )
        self.assertEqual(got, list(vector))

    def test_loader_ignores_a_sentinel_results_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "results.csv").write_text(
                "candidate_index,A_bending_capacity\n0,99\n",
                encoding="utf-8",
            )
            path = _write(directory, _payload(vectors=[_VECTOR_A]))
            loaded = load_initial_design(
                path,
                algorithm_name="SCIPY_COBYLA_WARM_START",
                problem_id="L8100_B2",
            )
        self.assertEqual(loaded, (tuple(_VECTOR_A),))
