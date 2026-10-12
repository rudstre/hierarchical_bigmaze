"""Route maps from saved low-rank LMDP-HMM predictor bundles.

The quantities reproduce Qin's ``LowRankLMDP_HMM`` in validation mode
(``calculate_transition_matrix_and_policies(validation=True)``):

    occupancy[r] = normalize_L1(sigmoid(R[r]) * mask)
    route_weights = normalize_L1(sigmoid(pi))

where ``mask`` marks location-actions with at least one outgoing edge. The last
entry of ``route_weights`` belongs to the extra uniform route the model
appends (its occupancy is the normalised mask), so it is reported separately.

Qin's state index is ``i = a * 49 + s`` with ``(x, y) = (s // 7, s % 7)`` and
actions ordered (right, up, left, down).

How a route enters the model determines what is worth plotting. The passive
transition from one location-action to the next is proportional to the
route's occupancy of the next location-action, normalised over the adjacent
options, and multiplied by the target route's weight. Hence:

- within a route, only ratios between actions at the same location matter,
  i.e. ``action_given_location[r, s, a] = occupancy[r, s, a] / sum_a``;
- a location's total mass ``location_mass[r, s]`` only matters when routes are
  compared (route inference and switching);
- a route with weight ~0 can never be entered, so its ``R`` does not affect
  the likelihood.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import expit

from andrew_mlmdp.regression_execution import _prepare_qin_imports

N_LOCATIONS = 49
N_ACTIONS = 4


def _decode_tensor(record):
    values = np.asarray(record["values"], dtype=np.float64)
    if list(values.shape) != list(record["shape"]):
        raise ValueError("HMM state shape does not match its recorded shape")
    return values


def hmm_route_maps(bundle, project_root: str | Path) -> dict[str, object]:
    """Per-route location-action occupancy and route weights for one partition."""
    if bundle.get("status") != "success":
        raise ValueError(
            f"Predictor bundle status is {bundle.get('status')!r}, not 'success'"
        )
    route = bundle["route"]
    if route["family"] != "hmm":
        raise ValueError(f"Route family is {route['family']!r}, not 'hmm'")
    n_routes = int(route["settings"]["n_routes"])
    R = _decode_tensor(route["state_dict"]["R"])
    pi = _decode_tensor(route["state_dict"]["pi"])
    if R.shape != (n_routes, N_ACTIONS * N_LOCATIONS) or pi.shape != (n_routes + 1,):
        raise ValueError(
            f"Unexpected HMM parameter shapes R{R.shape}, pi{pi.shape} "
            f"for {n_routes} routes"
        )

    _prepare_qin_imports(Path(project_root))
    from mazehelper.transition_matrix_functions import (
        location_action_adjacency_matrix_from_maze_id,
    )

    maze_id = int(bundle["partition"]["maze_id"])
    adjacency = location_action_adjacency_matrix_from_maze_id(maze_id)
    mask = (adjacency.sum(dim=-1) > 0).numpy().astype(np.float64)

    occupancy = expit(R) * mask
    occupancy /= occupancy.sum(axis=-1, keepdims=True)
    weights = expit(pi)
    weights /= weights.sum()
    # (route, location, action) from Qin's action-major state index.
    occupancy = occupancy.reshape(n_routes, N_ACTIONS, N_LOCATIONS).transpose(0, 2, 1)
    location_mass = occupancy.sum(axis=-1)
    # Locations without valid actions keep exact zeros rather than 0/0.
    action_given_location = np.divide(
        occupancy,
        location_mass[..., None],
        out=np.zeros_like(occupancy),
        where=location_mass[..., None] > 0,
    )
    return {
        "maze_id": maze_id,
        "occupancy": occupancy,
        "location_mass": location_mass,
        "action_given_location": action_given_location,
        "route_logits": pi[:n_routes],
        "route_weights": weights[:n_routes],
        "uniform_route_weight": float(weights[n_routes]),
        "partition": bundle["partition"],
        "artifact_digest": bundle.get("artifact_digest"),
        "seed": route.get("seed"),
        "settings": route["settings"],
    }


def make_hmm_route_figure(
    maps, project_root: str | Path, *, routes: list[int] | None = None, n_columns=5
):
    """One maze panel per selected route (default: all), on Qin's maze drawing.

    Arrow length is ``action_given_location``: the route's preference among the
    actions at each location, which is what shapes action choice within the
    route. Colour is ``location_mass`` on a scale shared by the plotted panels,
    so routes can be compared where it matters (route inference and switching).
    """
    import matplotlib.pyplot as plt

    _prepare_qin_imports(Path(project_root))
    from mazehelper.plotting_functions import plot_maze

    # Location s sits at (x, y) = (s // 7, s % 7); action unit steps in Qin's
    # (right, up, left, down) order. Arrows are drawn in one quiver call so all
    # actions share a length scale (Qin's plot_policy_all autoscales each
    # direction separately).
    x = np.repeat(np.arange(7), 7)
    y = np.tile(np.arange(7), 7)
    step_x = np.array([1, 0, -1, 0])
    step_y = np.array([0, 1, 0, -1])

    if routes is None:
        routes = list(range(maps["occupancy"].shape[0]))
    if not routes:
        raise ValueError("No routes selected for plotting")
    mass_max = maps["location_mass"][routes].max()
    n_columns = min(n_columns, len(routes))
    n_rows = -(-len(routes) // n_columns)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(2.6 * n_columns, 2.8 * n_rows), squeeze=False
    )
    image = None
    for panel, ax in enumerate(axes.flat):
        if panel >= len(routes):
            ax.set_visible(False)
            continue
        route_index = routes[panel]
        preference = maps["action_given_location"][route_index]
        # location_mass[s] -> [x, y]; imshow wants rows = y, columns = x.
        image = ax.imshow(
            maps["location_mass"][route_index].reshape(7, 7).T,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=mass_max,
        )
        plot_maze(
            maps["maze_id"],
            ax=ax,
            fig=fig,
            color="grey",
            transition_linewidth=3,
            node_width=30,
        )
        ax.quiver(
            np.tile(x, N_ACTIONS),
            np.tile(y, N_ACTIONS),
            (preference * step_x).T.ravel(),
            (preference * step_y).T.ravel(),
            # p(a | location) = 1 spans half an edge, so arrows leaving
            # neighbouring nodes along the same corridor do not overlap.
            color="white",
            edgecolor="black",
            linewidth=0.4,
            angles="xy",
            scale_units="xy",
            scale=2.2,
            width=0.012,
            minlength=0,
            zorder=2,
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Route {route_index} · weight {maps['route_weights'][route_index]:.3f}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.6, label="Location mass under route"
    )
    return fig
