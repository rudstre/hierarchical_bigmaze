import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_rank_validation import _config, _successful_shard

import andrew_mlmdp.validation as validation
import andrew_mlmdp.validation_aggregation as aggregation


def _context(config, *, source_sha: str):
    return SimpleNamespace(
        compatibility={
            "sweep_signature": config.sweep_signature,
            "data_sha256": "data",
            "maze_sha256": "maze",
            "source": {
                "scope": "worker_and_model_source",
                "git_head": "head",
                "content_sha256": source_sha,
            },
            "runtime": {"python": "test"},
        }
    )


def test_presentation_aggregation_accepts_stored_worker_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    stored = _context(config, source_sha="worker-at-submission")
    current = _context(config, source_sha="worker-after-aggregation-edit")
    monkeypatch.setattr(aggregation, "_load_problem_context", lambda _: current)
    shards = tmp_path / "shards"
    shards.mkdir()
    shard = _successful_shard(config, stored.compatibility, 2, -4.0)
    (shards / "k_02.json").write_text(json.dumps(shard))

    result = aggregation.aggregate_rank_results(
        config,
        shards,
        tmp_path / "aggregate",
    )

    assert result["best_k"] == 2
    assert result["worker_compatibility"] == stored.compatibility
    assert result["aggregation_source"]["content_sha256"]
    assert (tmp_path / "aggregate" / "held_out_log_likelihood_vs_k.png").is_file()
    assert (tmp_path / "aggregate" / "held_out_log_likelihood_vs_k.svg").is_file()


def test_presentation_aggregation_rejects_mixed_worker_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    first_context = _context(config, source_sha="worker-one")
    second_context = _context(config, source_sha="worker-two")
    monkeypatch.setattr(
        aggregation,
        "_load_problem_context",
        lambda _: first_context,
    )
    shards = tmp_path / "shards"
    shards.mkdir()
    first = _successful_shard(config, first_context.compatibility, 2, -4.0)
    second = _successful_shard(config, second_context.compatibility, 3, -3.0)
    (shards / "k_02.json").write_text(json.dumps(first))
    (shards / "k_03.json").write_text(json.dumps(second))

    with pytest.raises(ValueError, match="different worker, data, or config"):
        aggregation.aggregate_rank_results(config, shards, tmp_path / "aggregate")


def test_worker_fingerprint_excludes_only_aggregation_sources(tmp_path: Path):
    (tmp_path / "src" / "andrew_mlmdp").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    worker = tmp_path / "src" / "andrew_mlmdp" / "model.py"
    worker.write_text("VALUE = 1\n")
    aggregate_module = tmp_path / "src" / "andrew_mlmdp" / "validation_aggregation.py"
    aggregate_module.write_text("PLOT = 1\n")
    aggregate_cli = tmp_path / "scripts" / "aggregate_hierarchy_rank_validation.py"
    aggregate_cli.write_text("print('one')\n")
    first = validation.source_code_fingerprint(tmp_path)

    aggregate_module.write_text("PLOT = 2\n")
    aggregate_cli.write_text("print('two')\n")
    presentation_edit = validation.source_code_fingerprint(tmp_path)
    worker.write_text("VALUE = 2\n")
    worker_edit = validation.source_code_fingerprint(tmp_path)

    assert first["scope"] == "worker_and_model_source"
    assert first["content_sha256"] == presentation_edit["content_sha256"]
    assert presentation_edit["content_sha256"] != worker_edit["content_sha256"]
