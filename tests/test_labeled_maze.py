import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from andrew_mlmdp import (
    Environment,
    load_doohan_maze,
    maze_from_labeled_edges,
)

DOOHAN_CONFIG = (
    Path(__file__).parents[1]
    / "external"
    / "GridMaze-mFC-ephys-DATA"
    / "data"
    / "experiment_info"
    / "maze_configs.json"
)


def test_labeled_edges_create_tower_states_with_explicit_connections() -> None:
    definition = maze_from_labeled_edges(
        ["A1-A2", "A2-B2"],
        tower_shape=(2, 2),
    )

    assert definition.maze.shape == (2, 2)
    assert len(definition.maze.free_cells) == 4
    assert definition.coordinate_for("A1") == (1, 0)
    assert definition.coordinate_for("B2") == (0, 1)
    assert definition.label_for((0, 0)) == "A2"
    with pytest.raises(ValueError, match="Unknown maze label"):
        definition.coordinate_for("A1-A2")


def test_labeled_maze_mappings_are_validated_and_immutable() -> None:
    definition = maze_from_labeled_edges(["A1-A2"], tower_shape=(2, 1))

    with pytest.raises(TypeError):
        cast(dict[str, tuple[int, int]], definition.coordinate_by_label)[
            "new"
        ] = (0, 0)
    with pytest.raises(ValueError, match="Unknown maze label"):
        definition.coordinate_for("G7")
    with pytest.raises(ValueError, match="not a maze state"):
        definition.label_for((1, 1))


def test_missing_edges_are_blocked_under_existing_command_dynamics() -> None:
    definition = maze_from_labeled_edges(["A1-A2"], tower_shape=(2, 2))
    maze = definition.maze
    a1 = definition.coordinate_for("A1")

    assert maze.command_outcome(a1, "north") == definition.coordinate_for("A2")
    assert maze.command_outcome(a1, "east") == a1


@pytest.mark.parametrize(
    ("edges", "tower_shape", "message"),
    [
        (["A1/A2"], (2, 2), "Malformed"),
        (["A1-A1"], (2, 2), "itself"),
        (["A1-B2"], (2, 2), "cardinally adjacent"),
        (["A1-C1"], (2, 2), "outside"),
        (["A1-A2", "A2-A1"], (2, 2), "Duplicate"),
    ],
)
def test_invalid_edge_definitions_are_rejected(edges, tower_shape, message) -> None:
    with pytest.raises(ValueError, match=message):
        maze_from_labeled_edges(edges, tower_shape=tower_shape)


@pytest.mark.parametrize("tower_shape", [(0, 2), (2, -1), (2,), [2, 2], (2, 27)])
def test_invalid_tower_shapes_are_rejected(tower_shape) -> None:
    with pytest.raises(ValueError, match="Tower shape|26"):
        maze_from_labeled_edges([], tower_shape=tower_shape)


def test_doohan_loader_reads_an_explicit_configuration(tmp_path) -> None:
    config_path = tmp_path / "maze_configs.json"
    config_path.write_text(
        json.dumps({"tiny": {"structure": ["A1-A2"]}}),
        encoding="utf-8",
    )

    definition = load_doohan_maze("tiny", config_path)

    assert definition.coordinate_for("A1") == (6, 0)
    with pytest.raises(ValueError, match="Unknown Doohan maze"):
        load_doohan_maze("missing", config_path)
    with pytest.raises(FileNotFoundError):
        load_doohan_maze("tiny", tmp_path / "missing.json")


@pytest.mark.skipif(
    not DOOHAN_CONFIG.is_file(),
    reason="Doohan experiment_info data has not been downloaded",
)
@pytest.mark.parametrize(
    ("maze_name", "n_states"),
    [("maze_1", 49), ("maze_2", 49), ("rooms_maze", 49)],
)
def test_downloaded_doohan_mazes_are_connected(maze_name, n_states) -> None:
    definition = load_doohan_maze(maze_name)
    maze = definition.maze

    assert maze.shape == (7, 7)
    assert len(maze.free_cells) == n_states
    assert definition.coordinate_for("A1") == (6, 0)
    assert definition.coordinate_for("G7") == (0, 6)
    assert maze.reachable_cells(definition.coordinate_for("A1")) == set(
        maze.free_cells
    )


@pytest.mark.skipif(
    not DOOHAN_CONFIG.is_file(),
    reason="Doohan experiment_info data has not been downloaded",
)
def test_downloaded_doohan_maze_solves_as_an_lmdp() -> None:
    definition = load_doohan_maze("maze_1")
    environment = Environment(definition.maze)
    solution = environment.solve(definition.coordinate_for("G7"))

    assert np.allclose(environment.passive.sum(axis=0), 1.0)
    assert np.allclose(solution.controlled.sum(axis=0), 1.0)
