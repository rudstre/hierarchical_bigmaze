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
from andrew_mlmdp.validation import AdamValidationConfig, RankValidationError


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


def _fake_discovery_config(maze_name="maze_1"):
    """A stand-in for a loaded RankValidationConfig, for monkeypatching
    load_validation_config in tests that exercise load_adjacent_regression_config."""

    return SimpleNamespace(
        dataset=SimpleNamespace(maze_name=maze_name),
        discovery=SimpleNamespace(),
        marker=f"discovery-config-{maze_name}",
    )


def test_adjacent_config_allows_reduced_test_rank_grid(tmp_path):
    config = AdjacentRegressionConfig(
        dataset=AdjacentDatasetConfig("data", ("m2",)),
        discovery_config="base.json",
        discovery_dir="discovery",
        adam=AdamValidationConfig(),
        ranks=(2, 5, 10),
        project_root=tmp_path,
    )

    assert config.ranks == (2, 5, 10)


def test_adjacent_config_loads_inclusive_rank_bounds(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    monkeypatch.setattr(
        adjacent, "load_validation_config", lambda path: _fake_discovery_config()
    )
    config_path = tmp_path / "adjacent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {
                    "data_root": "data",
                    "subject_ids": ["m2"],
                    "maze_name": "maze_1",
                    "start_date": None,
                    "end_date": None,
                },
                "discovery_config": "base.json",
                "discovery_dir": "discovery",
                "adam": {},
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


def test_adjacent_config_rejects_mismatched_discovery_maze(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    monkeypatch.setattr(
        adjacent,
        "load_validation_config",
        lambda path: _fake_discovery_config("maze_2"),
    )
    config_path = tmp_path / "adjacent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {
                    "data_root": "data",
                    "subject_ids": ["m2"],
                    "maze_name": "maze_1",
                    "start_date": None,
                    "end_date": None,
                },
                "discovery_config": "base.json",
                "discovery_dir": "discovery",
                "adam": {},
                "ranks": [2],
            }
        )
    )

    with pytest.raises(ValueError, match="maze_name"):
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
                ],
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
        discovery_config="base.json",
        discovery_dir="discovery",
        adam=AdamValidationConfig(),
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
        adjacent, "load_validation_config", lambda *a: _fake_discovery_config()
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


def test_adjacent_config_discovery_dir_defaults_when_omitted(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    config_path = tmp_path / "adjacent.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {
                    "data_root": "data",
                    "subject_ids": ["m2"],
                    "maze_name": "maze_1",
                    "start_date": None,
                    "end_date": None,
                },
                "discovery_config": "base.json",
                "adam": {},
                "rank_min": 2,
                "rank_max": 3,
            }
        )
    )
    monkeypatch.setattr(
        adjacent,
        "load_validation_config",
        lambda path: SimpleNamespace(
            marker=str(path), dataset=SimpleNamespace(maze_name="maze_1")
        ),
    )
    monkeypatch.setattr(
        adjacent, "_load_dataset_context", lambda base: SimpleNamespace()
    )
    monkeypatch.setattr(
        adjacent, "_discovery_compatibility", lambda base, ctx: {"marker": base.marker}
    )
    monkeypatch.setattr(adjacent, "_payload_digest", lambda payload: "deadbeef")

    config = load_adjacent_regression_config(config_path)

    expected = (tmp_path / "data" / "nmf_bases" / "maze_1" / "deadbeef").resolve()
    assert config.resolved_discovery_dir == expected

    config_again = load_adjacent_regression_config(config_path)
    assert config_again.resolved_discovery_dir == config.resolved_discovery_dir


def test_adjacent_config_slurm_section_tolerated_and_ignored_by_signature(
    tmp_path, monkeypatch
):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    monkeypatch.setattr(
        adjacent, "load_validation_config", lambda path: _fake_discovery_config()
    )
    base_payload = {
        "schema_version": 2,
        "dataset": {
            "data_root": "data",
            "subject_ids": ["m2"],
            "maze_name": "maze_1",
            "start_date": None,
            "end_date": None,
        },
        "discovery_config": "base.json",
        "discovery_dir": "discovery",
        "adam": {},
        "ranks": [2, 5],
    }
    plain_path = tmp_path / "plain.json"
    plain_path.write_text(json.dumps(base_payload))
    with_slurm_path = tmp_path / "with_slurm.json"
    with_slurm_path.write_text(
        json.dumps({**base_payload, "slurm": {"partition": "gpu", "bands": []}})
    )

    plain = load_adjacent_regression_config(plain_path)
    with_slurm = load_adjacent_regression_config(with_slurm_path)

    assert with_slurm.ranks == plain.ranks
    assert with_slurm.signature == plain.signature


def test_aggregate_outer_fold_exclude_ranks_prevents_permanent_pending(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    config = AdjacentRegressionConfig(
        dataset=AdjacentDatasetConfig("data", ("m2",)),
        discovery_config="base.json",
        discovery_dir="discovery",
        adam=AdamValidationConfig(),
        ranks=(2, 3),
        project_root=tmp_path,
    )
    identity = _worker_identity()
    digest = "fold-digest"
    sessions = list(identity["route_training_session_ids"])
    fold_record = {
        "fold_identity_digest": digest,
        "fold_identity": identity,
        "inner_validation_session_ids": sessions,
    }
    output = tmp_path / "output"
    source = adjacent.source_code_fingerprint(
        config.project_root, config_path=config.source_path
    )
    route_sessions = tuple(identity["route_training_session_ids"])
    for session in sessions:
        training_sessions = tuple(value for value in route_sessions if value != session)
        compatibility = adjacent._inner_compatibility(
            config, identity, digest, training_sessions, session, 2, source=source
        )
        shard = adjacent._inner_shard_path(output, digest, 2, session)
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(
            json.dumps(
                {
                    "schema_version": adjacent.ADJACENT_SCHEMA_VERSION,
                    "artifact_type": "adjacent_mlmdp_inner_fit",
                    "status": "success",
                    "k": 2,
                    "compatibility": compatibility,
                    "validation_session_id": session,
                    "validation_ll_per_transition": -1.0,
                }
            )
        )
    # No shards at all for rank 3 -- simulates a rank whose NMF discovery
    # scientifically failed and will never be submitted.

    pending = adjacent.aggregate_outer_fold(config, output, fold_record=fold_record)
    assert pending["status"] == "pending"

    selected = adjacent.aggregate_outer_fold(
        config, output, fold_record=fold_record, exclude_ranks=frozenset({3})
    )
    assert selected["status"] == "selected"
    assert selected["selection"]["selected_k"] == 2


def test_aggregate_outer_fold_exclude_all_ranks_raises(tmp_path):
    config = _worker_config(tmp_path)
    fold_record = {
        "fold_identity_digest": "fold",
        "fold_identity": _worker_identity(),
        "inner_validation_session_ids": ["route-a"],
    }
    with pytest.raises(ValueError, match="No ranks remain eligible"):
        adjacent.aggregate_outer_fold(
            config,
            tmp_path / "output",
            fold_record=fold_record,
            exclude_ranks=frozenset({2}),
        )


def test_inner_input_failure_remains_operational(tmp_path, monkeypatch):
    config = _worker_config(tmp_path)
    monkeypatch.setattr(adjacent, "load_adjacent_dataset", lambda *a, **k: object())
    monkeypatch.setattr(adjacent, "_trials_for_sessions", lambda *a, **k: (object(),))
    monkeypatch.setattr(
        adjacent, "load_validation_config", lambda *a: _fake_discovery_config()
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
