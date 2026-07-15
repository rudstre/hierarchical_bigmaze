"""Grid mazes used by the MLMDP navigation experiments.

Coordinates are always ``(row, column)`` with ``(0, 0)`` at the top left.
Free cells receive integer state IDs in row-major order. These conventions are
kept explicit because later transition matrices depend on the state ordering.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Literal, TypeAlias


Coordinate: TypeAlias = tuple[int, int]
Command: TypeAlias = Literal["north", "south", "east", "west", "stay"]

COMMAND_DELTAS: dict[Command, Coordinate] = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
    "stay": (0, 0),
}


@dataclass(frozen=True)
class Maze:
    """A parsed grid maze with stable physical-state numbering."""

    ascii_rows: tuple[str, ...]
    free_cells: tuple[Coordinate, ...]
    walls: frozenset[Coordinate]
    state_by_coordinate: dict[Coordinate, int]

    @classmethod
    def from_ascii(cls, layout: str) -> "Maze":
        """Parse a maze containing ``#`` walls and ``.`` free cells."""

        rows = _normalise_layout(layout)

        free_cells: list[Coordinate] = []
        walls: set[Coordinate] = set()

        for row, line in enumerate(rows):
            for column, symbol in enumerate(line):
                coordinate = (row, column)

                if symbol not in {"#", "."}:
                    raise ValueError(
                        f"Unknown maze symbol {symbol!r} at {coordinate}"
                    )

                if symbol == "#":
                    walls.add(coordinate)
                    continue

                free_cells.append(coordinate)

        state_by_coordinate: dict[Coordinate, int] = {}
        for state, coordinate in enumerate(free_cells):
            state_by_coordinate[coordinate] = state

        return cls(
            ascii_rows=rows,
            free_cells=tuple(free_cells),
            walls=frozenset(walls),
            state_by_coordinate=state_by_coordinate,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Maze":
        """Load maze geometry from a UTF-8 text file."""

        maze_text = Path(path).read_text(encoding="utf-8")
        return cls.from_ascii(maze_text)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(number of rows, number of columns)``."""

        return (len(self.ascii_rows), len(self.ascii_rows[0]))

    def state_index(self, coordinate: Coordinate) -> int:
        """Return the row-major state ID for a free coordinate."""

        try:
            return self.state_by_coordinate[coordinate]
        except KeyError as error:
            raise ValueError(f"Coordinate {coordinate} is not a free cell") from error

    def coordinate(self, state_index: int) -> Coordinate:
        """Return the coordinate associated with a physical-state ID."""

        if state_index < 0 or state_index >= len(self.free_cells):
            raise ValueError(f"State index {state_index} is out of range")
        return self.free_cells[state_index]

    def is_free(self, coordinate: Coordinate) -> bool:
        """Return whether a coordinate is a traversable physical cell."""

        return coordinate in self.state_by_coordinate

    def command_outcome(self, coordinate: Coordinate, command: Command) -> Coordinate:
        """Apply one grid command, returning the current cell if movement fails."""

        if not self.is_free(coordinate):
            raise ValueError(f"Coordinate {coordinate} is not a free cell")
        if command not in COMMAND_DELTAS:
            raise ValueError(f"Unknown movement command {command!r}")

        row_change, column_change = COMMAND_DELTAS[command]
        next_coordinate = (
            coordinate[0] + row_change,
            coordinate[1] + column_change,
        )

        # Invalid commands become self-transitions in the passive dynamics.
        if not self.is_free(next_coordinate):
            return coordinate
        return next_coordinate

    def reachable_cells(self, start: Coordinate) -> set[Coordinate]:
        """Return free cells connected to ``start`` by cardinal movement."""

        if not self.is_free(start):
            raise ValueError(f"Coordinate {start} is not a free cell")

        reached = {start}
        cells_to_visit = deque([start])

        while cells_to_visit:
            coordinate = cells_to_visit.popleft()
            for command in ("north", "south", "east", "west"):
                neighbour = self.command_outcome(coordinate, command)
                if neighbour not in reached:
                    reached.add(neighbour)
                    cells_to_visit.append(neighbour)

        return reached

    def to_ascii(self) -> str:
        """Return the normalized source layout."""

        return "\n".join(self.ascii_rows)


def _normalise_layout(layout: str) -> tuple[str, ...]:
    normalized = dedent(layout).strip("\n")
    if not normalized:
        raise ValueError("Maze layout is empty")

    rows = tuple(normalized.splitlines())
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("Maze layout must be rectangular")
    return rows

