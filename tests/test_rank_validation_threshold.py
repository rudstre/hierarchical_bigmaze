from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_rank_validation import _config

import andrew_mlmdp.lmdp as lmdp
import andrew_mlmdp.validation as validation
from andrew_mlmdp.hierarchy.model import SubgoalBasis
from andrew_mlmdp.validation import AdamValidationConfig


class _FakeEnvironment:
    maze = object()

    def __init__(self, threshold_cap: float):
        self.threshold_cap = threshold_cap
        self.calls = []

    def hierarchy(
        self,
        basis,
        *,
        parameters,
        goal_reward_mode="inpainted",
        commitment_mode="termination",
        commitment_radius=None,
    ):
        self.calls.append(
            (basis, parameters, goal_reward_mode, commitment_mode, commitment_radius)
        )
        return SimpleNamespace(
            threshold_range=lambda goals: SimpleNamespace(
                maximum=self.threshold_cap,
                limiting_pairs=(((0, 1), 0),),
            )
        )


def _rank_result():
    return SimpleNamespace(
        discovery=SimpleNamespace(profiles=np.ones((2, 2), dtype=np.float64))
    )


def _install_template_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SubgoalBasis,
        "from_profiles",
        lambda maze, profiles, **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        lmdp,
        "soft_parameters",
        lambda k, **values: {"k": k, **values},
    )


def test_initial_template_resolves_fraction_against_small_rank_cap(
    tmp_path,
    monkeypatch,
):
    _install_template_fakes(monkeypatch)
    config = _config(tmp_path)
    environment = _FakeEnvironment(threshold_cap=1e-6)

    template, domain, values = validation._initial_template(
        environment,
        _rank_result(),
        config,
        49,
        {(0, 1)},
    )

    assert domain.maximum == pytest.approx(1e-6)
    assert values["core_threshold"] == pytest.approx(4e-7)
    assert 0.0 < values["core_threshold"] < domain.maximum
    assert environment.calls[0][0].core_threshold == 0.0
    assert environment.calls[1][0].core_threshold == pytest.approx(4e-7)
    assert template is not None
    # Default config: both hierarchy() calls use the unmodified defaults.
    for call in environment.calls:
        assert call[2:] == ("inpainted", "termination", None)


def test_initial_template_threads_goal_reward_mode_and_commitment_settings(
    tmp_path,
    monkeypatch,
):
    _install_template_fakes(monkeypatch)
    config = replace(
        _config(tmp_path),
        adam=AdamValidationConfig(
            goal_reward_mode="deferred",
            commitment_mode="radius",
            commitment_radius=2,
        ),
    )
    environment = _FakeEnvironment(threshold_cap=1e-1)

    validation._initial_template(
        environment,
        _rank_result(),
        config,
        49,
        {(0, 1)},
    )

    assert len(environment.calls) == 2
    for call in environment.calls:
        assert call[2:] == ("deferred", "radius", 2)


def test_initial_template_rejects_unrepresentable_threshold_domain(
    tmp_path,
    monkeypatch,
):
    _install_template_fakes(monkeypatch)
    config = _config(tmp_path)
    environment = _FakeEnvironment(threshold_cap=np.finfo(np.float64).eps)

    with pytest.raises(ValueError, match="no representable initial value"):
        validation._initial_template(
            environment,
            _rank_result(),
            config,
            49,
            {(0, 1)},
        )
