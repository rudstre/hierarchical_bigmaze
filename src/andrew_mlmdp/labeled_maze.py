"""Adapters for labeled edge-list mazes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from andrew_mlmdp.maze import Coordinate, Maze

_TOWER_LABEL = re.compile(r"([A-Z])([1-9][0-9]*)")
_DEFAULT_DOOHAN_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "GridMaze-mFC-ephys-DATA"
    / "data"
    / "experiment_info"
    / "maze_configs.json"
)


@dataclass(frozen=True)
class LabeledMaze:
    """A raster maze with stable labels for every traversable state."""

    maze: Maze
    coordinate_by_label: Mapping[str, Coordinate]
    label_by_coordinate: Mapping[Coordinate, str]

    def __post_init__(self) -> None:
        coordinate_by_label = dict(self.coordinate_by_label)
        label_by_coordinate = dict(self.label_by_coordinate)
        expected_reverse = {
            coordinate: label for label, coordinate in coordinate_by_label.items()
        }

        if len(expected_reverse) != len(coordinate_by_label):
            raise ValueError("Maze labels must map to unique coordinates")
        if label_by_coordinate != expected_reverse:
            raise ValueError("Label mappings must be exact inverses")
        if set(label_by_coordinate) != set(self.maze.free_cells):
            raise ValueError("Every free maze state must have exactly one label")

        object.__setattr__(
            self,
            "coordinate_by_label",
            MappingProxyType(coordinate_by_label),
        )
        object.__setattr__(
            self,
            "label_by_coordinate",
            MappingProxyType(label_by_coordinate),
        )

    def coordinate_for(self, label: str) -> Coordinate:
        """Return the raster coordinate for a tower or bridge label."""

        try:
            return self.coordinate_by_label[label]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unknown maze label {label!r}") from error

    def label_for(self, coordinate: Coordinate) -> str:
        """Return the tower or bridge label for a raster coordinate."""

        try:
            return self.label_by_coordinate[coordinate]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Coordinate {coordinate!r} is not a maze state"
            ) from error


def maze_from_labeled_edges(
    edges: Iterable[str],
    *,
    tower_shape: tuple[int, int] = (7, 7),
) -> LabeledMaze:
    """Expand a labeled tower graph into a raster of tower and bridge states."""

    number_of_rows, number_of_columns = _validated_tower_shape(tower_shape)
    if isinstance(edges, (str, bytes)):
        raise ValueError("Maze edges must be an iterable of edge labels")

    coordinate_by_label = {
        f"{chr(ord('A') + column)}{number}": (
            2 * (number_of_rows - number),
            2 * column,
        )
        for column in range(number_of_columns)
        for number in range(1, number_of_rows + 1)
    }
    seen_edges: set[frozenset[str]] = set()

    for edge in edges:
        start_label, end_label = _parse_edge_label(edge)
        try:
            start = coordinate_by_label[start_label]
            end = coordinate_by_label[end_label]
        except KeyError as error:
            raise ValueError(
                f"Edge {edge!r} contains a tower outside {tower_shape}"
            ) from error

        edge_key = frozenset((start_label, end_label))
        if start_label == end_label:
            raise ValueError(f"Edge {edge!r} cannot connect a tower to itself")
        if edge_key in seen_edges:
            raise ValueError(f"Duplicate maze edge {edge!r}")
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) != 2:
            raise ValueError(
                f"Edge {edge!r} must connect cardinally adjacent towers"
            )

        seen_edges.add(edge_key)
        bridge = (
            (start[0] + end[0]) // 2,
            (start[1] + end[1]) // 2,
        )
        coordinate_by_label[edge] = bridge

    raster_shape = (2 * number_of_rows - 1, 2 * number_of_columns - 1)
    free_coordinates = set(coordinate_by_label.values())
    layout = "\n".join(
        "".join(
            "." if (row, column) in free_coordinates else "#"
            for column in range(raster_shape[1])
        )
        for row in range(raster_shape[0])
    )
    maze = Maze.from_ascii(layout)
    label_by_coordinate = {
        coordinate: label for label, coordinate in coordinate_by_label.items()
    }
    return LabeledMaze(
        maze=maze,
        coordinate_by_label=coordinate_by_label,
        label_by_coordinate=label_by_coordinate,
    )


def load_doohan_maze(
    maze_name: str,
    config_path: str | Path | None = None,
) -> LabeledMaze:
    """Load one Doohan maze from ``maze_configs.json``."""

    path = _DEFAULT_DOOHAN_CONFIG if config_path is None else Path(config_path)
    with path.open(encoding="utf-8") as input_file:
        configurations = json.load(input_file)

    if not isinstance(configurations, dict):
        raise ValueError("Maze configuration must contain a JSON object")
    try:
        configuration = configurations[maze_name]
    except (KeyError, TypeError) as error:
        available = ", ".join(sorted(configurations))
        raise ValueError(
            f"Unknown Doohan maze {maze_name!r}; available mazes: {available}"
        ) from error
    if not isinstance(configuration, dict) or not isinstance(
        configuration.get("structure"),
        list,
    ):
        raise ValueError(f"Maze {maze_name!r} has no valid structure edge list")

    return maze_from_labeled_edges(configuration["structure"])


def _validated_tower_shape(tower_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(tower_shape, tuple)
        or len(tower_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in tower_shape
        )
        or any(value <= 0 for value in tower_shape)
    ):
        raise ValueError("Tower shape must be a pair of positive integers")
    if tower_shape[1] > 26:
        raise ValueError("Labeled mazes support at most 26 tower columns")
    return tower_shape


def _parse_edge_label(edge: str) -> tuple[str, str]:
    if not isinstance(edge, str) or edge.count("-") != 1:
        raise ValueError(f"Malformed maze edge {edge!r}")
    start_label, end_label = edge.split("-")
    if not _TOWER_LABEL.fullmatch(start_label) or not _TOWER_LABEL.fullmatch(
        end_label
    ):
        raise ValueError(f"Malformed maze edge {edge!r}")
    return start_label, end_label
