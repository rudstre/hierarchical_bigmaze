"""Route maps decoded from saved HMM predictor bundles."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from andrew_mlmdp.hmm_route_plots import hmm_route_maps, make_hmm_route_figure
from andrew_mlmdp.regression_execution import _qin_route_imports

matplotlib.use("Agg")

ROOT = Path(__file__).parents[1]
MAZE_ID = 1
N_ROUTES = 3


def _qin_mask():
    from mazehelper.transition_matrix_functions import (
        location_action_adjacency_matrix_from_maze_id,
    )

    adjacency = location_action_adjacency_matrix_from_maze_id(MAZE_ID)
    return (adjacency.sum(dim=-1) > 0).numpy()


def _bundle(R, pi, *, settings=None):
    qin = _qin_route_imports(ROOT)
    settings = settings or qin["normalize_hmm"]({"n_routes": N_ROUTES})
    return {
        "status": "success",
        "artifact_digest": "digest",
        "partition": {"maze_id": MAZE_ID, "subject_id": "m0"},
        "route": {
            "family": "hmm",
            "settings": settings,
            "seed": 7,
            "state_dict": {
                name: {
                    "values": value.tolist(),
                    "shape": list(value.shape),
                    "dtype": "torch.float32",
                }
                for name, value in (("R", R), ("pi", pi))
            },
        },
    }


def _random_parameters(seed=0):
    _qin_route_imports(ROOT)
    rng = np.random.default_rng(seed)
    mask = _qin_mask()
    # As in the model's initialisation, masked entries are pushed to -1e10.
    R = np.where(mask, rng.normal(size=(N_ROUTES, 196)), -1e10).astype(np.float32)
    pi = rng.normal(size=N_ROUTES + 1).astype(np.float32)
    return R, pi


def test_action_major_state_index_maps_to_location_and_action():
    R, pi = _random_parameters()
    mask = _qin_mask()
    location, action = next(
        (s, a) for a in range(4) for s in range(49) if mask[a * 49 + s]
    )
    R[1] = np.where(mask, -30.0, -1e10)
    R[1, action * 49 + location] = 30.0
    maps = hmm_route_maps(_bundle(R, pi), ROOT)
    assert maps["occupancy"].shape == (N_ROUTES, 49, 4)
    assert np.unravel_index(maps["occupancy"][1].argmax(), (49, 4)) == (
        location,
        action,
    )


def test_occupancy_is_normalised_with_exact_zeros_on_impossible_actions():
    R, pi = _random_parameters()
    maps = hmm_route_maps(_bundle(R, pi), ROOT)
    impossible = ~_qin_mask().reshape(4, 49).T
    assert np.all(maps["occupancy"][:, impossible] == 0.0)
    np.testing.assert_allclose(maps["occupancy"].sum(axis=(1, 2)), 1.0)
    total = maps["route_weights"].sum() + maps["uniform_route_weight"]
    np.testing.assert_allclose(total, 1.0)


def test_action_preferences_are_normalised_within_each_location():
    R, pi = _random_parameters()
    maps = hmm_route_maps(_bundle(R, pi), ROOT)
    preference = maps["action_given_location"]
    has_actions = maps["location_mass"] > 0
    np.testing.assert_allclose(preference.sum(axis=-1)[has_actions], 1.0)
    assert np.all(preference[~has_actions] == 0.0)
    np.testing.assert_allclose(
        preference * maps["location_mass"][..., None], maps["occupancy"]
    )


def test_route_maps_match_qin_model_in_validation_mode():
    import torch

    qin = _qin_route_imports(ROOT)
    R, pi = _random_parameters(seed=3)
    bundle = _bundle(R, pi)
    model = qin["build_hmm"](MAZE_ID, bundle["route"]["settings"])
    model.load_state_dict({"R": torch.tensor(R), "pi": torch.tensor(pi)})
    model.calculate_transition_matrix_and_policies(validation=True)

    maps = hmm_route_maps(bundle, ROOT)
    expected = model.state_dist_given_route[:-1].detach().cpu().numpy()
    np.testing.assert_allclose(
        maps["occupancy"].transpose(0, 2, 1).reshape(N_ROUTES, 196),
        expected,
        rtol=1e-5,
        atol=1e-7,
    )
    weights = model.route_dist.detach().cpu().numpy()
    np.testing.assert_allclose(maps["route_weights"], weights[:-1], rtol=1e-5)
    np.testing.assert_allclose(maps["uniform_route_weight"], weights[-1], rtol=1e-5)


def test_non_hmm_or_failed_bundles_are_rejected():
    R, pi = _random_parameters()
    failed = {**_bundle(R, pi), "status": "scientific_failure"}
    with pytest.raises(ValueError, match="status"):
        hmm_route_maps(failed, ROOT)
    pca = _bundle(R, pi)
    pca["route"]["family"] = "pca"
    with pytest.raises(ValueError, match="family"):
        hmm_route_maps(pca, ROOT)


def test_figure_has_one_panel_per_selected_route():
    import matplotlib.pyplot as plt

    R, pi = _random_parameters()
    maps = hmm_route_maps(_bundle(R, pi), ROOT)
    for routes, expected in ((None, N_ROUTES), ([0, 2], 2)):
        figure = make_hmm_route_figure(maps, ROOT, routes=routes)
        titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        assert len(titles) == expected
        if routes:
            assert [t.split(" ")[1] for t in titles] == ["0", "2"]
        plt.close(figure)


def _panel_arrows(ax):
    from matplotlib.quiver import Quiver

    (quiver,) = [c for c in ax.collections if isinstance(c, Quiver)]
    return quiver.X, quiver.Y, quiver.U, quiver.V


def test_max_arrows_follow_qin_plot_policy_max():
    import matplotlib.pyplot as plt

    R, pi = _random_parameters(seed=5)
    maps = hmm_route_maps(_bundle(R, pi), ROOT)
    figure = make_hmm_route_figure(maps, ROOT, routes=[1], arrows="max")
    x, y, u, v = _panel_arrows(figure.axes[0])
    occupancy = maps["occupancy"][1]
    location_max = occupancy.max(axis=-1)
    reachable = location_max > 0
    # One arrow per location with any occupancy, in the argmax direction
    # (right, up, left, down), with length proportional to the max.
    assert len(x) == reachable.sum()
    locations = (np.asarray(x) * 7 + np.asarray(y)).astype(int)
    best = occupancy[locations].argmax(axis=-1)
    np.testing.assert_array_equal(np.sign(u), np.array([1, 0, -1, 0])[best])
    np.testing.assert_array_equal(np.sign(v), np.array([0, 1, 0, -1])[best])
    np.testing.assert_allclose(
        np.hypot(u, v), location_max[locations] / occupancy.max(), rtol=1e-6
    )
    plt.close(figure)


def test_all_and_conditional_arrows_scale_by_their_quantity():
    import matplotlib.pyplot as plt

    R, pi = _random_parameters(seed=6)
    maps = hmm_route_maps(_bundle(R, pi), ROOT)
    occupancy = maps["occupancy"][[0, 2]]
    expected = {
        "all": occupancy / occupancy.max(axis=(1, 2), keepdims=True),
        "conditional": maps["action_given_location"][[0, 2]],
    }
    for arrows, lengths in expected.items():
        figure = make_hmm_route_figure(maps, ROOT, routes=[0, 2], arrows=arrows)
        for panel in range(2):
            x, y, u, v = _panel_arrows(figure.axes[panel])
            # Impossible actions keep exact zeros and are not drawn.
            assert len(x) == (lengths[panel] > 0).sum()
            np.testing.assert_allclose(
                np.sort(np.hypot(u, v)),
                np.sort(lengths[panel][lengths[panel] > 0]),
                rtol=1e-6,
            )
        plt.close(figure)


def test_unknown_arrow_style_is_rejected():
    R, pi = _random_parameters()
    maps = hmm_route_maps(_bundle(R, pi), ROOT)
    with pytest.raises(ValueError, match="arrows"):
        make_hmm_route_figure(maps, ROOT, arrows="argmax")


def test_route_maps_report_fixed_and_learned_costs():
    R, pi = _random_parameters()
    fixed = hmm_route_maps(_bundle(R, pi), ROOT)["costs"]
    assert fixed["learned"] is False
    assert fixed["action_cost"] == pytest.approx(0.13)
    assert fixed["cognitive_constant"] == pytest.approx(30.0)

    bundle = _bundle(R, pi)
    # Learned costs are stored as log(a) and log(a * D).
    for name, value in (("log_action_cost", 0.2), ("log_switch_cost", 0.2 * 25.0)):
        bundle["route"]["state_dict"][name] = {
            "values": float(np.log(value)),
            "shape": [],
            "dtype": "torch.float32",
        }
    learned = hmm_route_maps(bundle, ROOT)["costs"]
    assert learned["learned"] is True
    assert learned["action_cost"] == pytest.approx(0.2)
    assert learned["cognitive_constant"] == pytest.approx(25.0)
