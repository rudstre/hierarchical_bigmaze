from pathlib import Path

import pytest

from andrew_mlmdp import Maze


FOUR_ROOMS_FILE = Path(__file__).parents[1] / "mazes" / "four_rooms.txt"


def test_four_rooms_fixture() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)

    assert maze.shape == (11, 11)
    assert len(maze.walls) == 24
    assert len(maze.free_cells) == 97


def test_state_ids_follow_row_major_order() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)

    for expected_state, coordinate in enumerate(maze.free_cells):
        assert maze.state_index(coordinate) == expected_state
        assert maze.coordinate(expected_state) == coordinate


def test_invalid_moves_return_the_current_cell() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)

    assert maze.command_outcome((0, 0), "north") == (0, 0)
    assert maze.command_outcome((0, 0), "west") == (0, 0)
    assert maze.command_outcome((0, 4), "east") == (0, 4)
    assert maze.command_outcome((0, 0), "stay") == (0, 0)
    assert maze.command_outcome((0, 0), "east") == (0, 1)


def test_ascii_round_trip() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    expected_layout = FOUR_ROOMS_FILE.read_text(encoding="utf-8").strip("\n")

    assert maze.to_ascii() == expected_layout


def test_all_free_cells_are_connected() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    reached = maze.reachable_cells((0, 0))

    assert reached == set(maze.free_cells)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ("", "empty"),
        ("..\n...", "rectangular"),
        (".x.", "Unknown maze symbol"),
    ],
)
def test_malformed_layouts_are_rejected(layout: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Maze.from_ascii(layout)
