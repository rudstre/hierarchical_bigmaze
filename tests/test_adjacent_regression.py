import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import andrew_mlmdp.adjacent_regression as adjacent
from andrew_mlmdp.adjacent_regression import (
    AdjacentDatasetConfig,
    AdjacentRegressionConfig,
    load_adjacent_regression_config,
    load_external_fold_predictors,
    run_inner_fit,
)
from andrew_mlmdp.validation import RankValidationError


@dataclass(frozen=True)
class _Identity:
    value: dict

    @property
    def digest(self):
        return hashlib.sha256(
            json.dumps(self.value, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def metadata(self):
        return self.value


@dataclass(frozen=True)
class _Fold:
    value: dict

    def identity(self, maze_id):
        assert maze_id == 1
        return _Identity(self.value)


def test_adjacent_config_allows_reduced_test_rank_grid(tmp_path):
    config = AdjacentRegressionConfig(
        dataset=AdjacentDatasetConfig("data", ("m2",)),
        base_validation_config="base.json",
        discovery_dir="discovery",
        ranks=(2, 5, 10),
        project_root=tmp_path,
    )

    assert config.ranks == (2, 5, 10)


def test_adjacent_config_loads_inclusive_rank_bounds(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    config_path = tmp_path / "adjacent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "data_root": "data",
                    "subject_ids": ["m2"],
                    "maze_name": "maze_1",
                    "start_date": None,
                    "end_date": None,
                },
                "base_validation_config": "base.json",
                "discovery_dir": "discovery",
                "rank_min": 3,
                "rank_max": 5,
            }
        )
    )

    config = load_adjacent_regression_config(config_path)

    assert config.ranks == (3, 4, 5)

    mixed = json.loads(config_path.read_text())
    mixed["ranks"] = [3, 5]
    config_path.write_text(json.dumps(mixed))
    with pytest.raises(ValueError, match="exactly one rank form"):
        load_adjacent_regression_config(config_path)


def test_inner_compatibility_accepts_precomputed_source(tmp_path):
    config = _worker_config(tmp_path)

    compatibility = adjacent._inner_compatibility(
        config,
        _worker_identity(),
        "fold",
        ("route-b", "route-c"),
        "route-a",
        2,
        source="fingerprint",
    )

    assert compatibility["source"] == "fingerprint"


def test_external_predictors_join_by_scientific_identity(tmp_path):
    identity = {
        "maze_id": 1,
        "subject_id": "m2",
        "regression_training_session_ids": ["train"],
        "validation_session_id": "validation",
        "route_training_session_ids": ["route-a", "route-b"],
    }
    fold = _Fold(identity)
    digest = fold.identity(1).digest
    root = tmp_path / "run"
    artifact_root = root / "folds" / digest
    artifact_root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "canonical_data_signature": "canonical",
                "folds": [
                    {
                        "fold_identity_digest": digest,
                        "fold_identity": identity,
                    }
                ]
            }
        )
    )
    rows = [
        {
            "subject_id": "m2",
            "session_id": "train",
            "trial_id": 1,
            "decision_order": 0,
            "action_0": 0.5,
            "action_1": 0.5,
            "action_2": 0.0,
            "action_3": 0.0,
        }
    ]
    (artifact_root / "predictor.json").write_text(
        json.dumps(
            {
                "status": "success",
                "fold_identity": identity,
                "prediction_columns": list(rows[0]),
                "prediction_rows": rows,
            }
        )
    )

    loaded = load_external_fold_predictors(
        root,
        folds=[fold],
        maze_id=1,
        canonical_signature="canonical",
    )

    assert set(loaded) == {digest}
    assert loaded[digest]["hierarchical_mlmdp"].action_0.tolist() == [0.5]

    with pytest.raises(ValueError, match="canonical data"):
        load_external_fold_predictors(
            root,
            folds=[fold],
            maze_id=1,
            canonical_signature="different",
        )

    bad_fold = _Fold({**identity, "validation_session_id": "different"})
    with pytest.raises(ValueError, match="fold identities"):
        load_external_fold_predictors(
            root,
            folds=[bad_fold],
            maze_id=1,
            canonical_signature="canonical",
        )


def _worker_config(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    return AdjacentRegressionConfig(
        dataset=AdjacentDatasetConfig("data", ("m2",)),
        base_validation_config="base.json",
        discovery_dir="discovery",
        ranks=(2,),
        project_root=tmp_path,
    )


def _worker_identity():
    return {
        "maze_id": 1,
        "subject_id": "m2",
        "regression_training_session_ids": ["regression"],
        "validation_session_id": "validation",
        "route_training_session_ids": ["route-a", "route-b", "route-c"],
    }


def test_inner_model_failure_is_terminal_scientific(tmp_path, monkeypatch):
    config = _worker_config(tmp_path)
    monkeypatch.setattr(adjacent, "load_adjacent_dataset", lambda *a, **k: object())
    monkeypatch.setattr(adjacent, "_trials_for_sessions", lambda *a, **k: (object(),))
    monkeypatch.setattr(
        adjacent, "load_validation_config", lambda *a: SimpleNamespace()
    )
    monkeypatch.setattr(
        adjacent,
        "_load_discovery_artifact",
        lambda *a: ({}, object(), "discovery"),
    )
    monkeypatch.setattr(adjacent, "_discovery_reference", lambda *a: {})
    monkeypatch.setattr(
        adjacent,
        "_fit_explicit_split",
        lambda *a: (_ for _ in ()).throw(RankValidationError("did not converge")),
    )

    result = run_inner_fit(
        config,
        tmp_path / "output",
        fold_identity=_worker_identity(),
        fold_identity_digest="fold",
        validation_session_id="route-a",
        k=2,
    )

    assert result["status"] == "scientific_failure"
    assert result["stage"] == "fit_and_score"


def test_inner_input_failure_remains_operational(tmp_path, monkeypatch):
    config = _worker_config(tmp_path)
    monkeypatch.setattr(adjacent, "load_adjacent_dataset", lambda *a, **k: object())
    monkeypatch.setattr(adjacent, "_trials_for_sessions", lambda *a, **k: (object(),))
    monkeypatch.setattr(
        adjacent, "load_validation_config", lambda *a: SimpleNamespace()
    )
    monkeypatch.setattr(
        adjacent,
        "_load_discovery_artifact",
        lambda *a: (_ for _ in ()).throw(ValueError("incompatible discovery")),
    )

    result = run_inner_fit(
        config,
        tmp_path / "output",
        fold_identity=_worker_identity(),
        fold_identity_digest="fold",
        validation_session_id="route-a",
        k=2,
    )

    assert result["status"] == "operational_failure"
    assert result["stage"] == "load_inputs"
