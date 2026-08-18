from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp import (
    LayerOneTaskLibrary,
    LMDPEnvironment,
    Maze,
    ModelParameters,
    SubgoalBasis,
)

PROJECT_ROOT = Path(__file__).parents[1]
FOUR_ROOM_SUBGOALS = (
    (0, 0),
    (9, 2),
    (2, 3),
    (3, 7),
    (9, 7),
    (7, 9),
)
FOUR_ROOM_GOAL = (10, 9)


@pytest.fixture(scope="session")
def four_room_environment() -> LMDPEnvironment:
    maze = Maze.from_file(PROJECT_ROOT / "mazes" / "four_rooms.txt")
    return LMDPEnvironment(maze)


@pytest.fixture(scope="session")
def regression_parameters() -> ModelParameters:
    return ModelParameters(
        interior_reward=-0.1,
        goal_reward=1.0,
        lower_control_cost=0.15,
        upper_control_cost=0.3,
        alpha=1.0,
        beta=10.0,
    )


@pytest.fixture(scope="session")
def four_room_template(
    four_room_environment,
    regression_parameters,
):
    basis = SubgoalBasis.from_locations(
        four_room_environment.maze,
        FOUR_ROOM_SUBGOALS,
        labels=("A", "B", "C", "D", "E", "F"),
    )
    return four_room_environment.hierarchy(
        basis,
        parameters=regression_parameters,
        task_library=LayerOneTaskLibrary.from_desirabilities(
            len(FOUR_ROOM_SUBGOALS),
            basis_target_desirability=np.exp(1.0 / 0.15),
            basis_off_target_desirability=np.exp(-2.0 / 0.15),
            basis_goal_desirability=np.exp(1.0 / 0.15),
        ),
    )


@pytest.fixture
def soft_corridor_template():
    """Frozen profiles isolate soft execution from NMF implementation changes."""

    maze = Maze.from_ascii("....\n....")
    environment = LMDPEnvironment(maze)
    profiles = np.asarray(
        [
            [1.0, 0.05],
            [0.9, 0.10],
            [0.3, 0.45],
            [0.1, 0.85],
            [0.8, 0.15],
            [0.5, 0.35],
            [0.15, 0.9],
            [0.05, 1.0],
        ],
        dtype=np.float64,
    )
    basis = SubgoalBasis.from_profiles(
        maze,
        profiles,
        core_threshold=0.25,
    )
    return environment.hierarchy(
        basis,
        parameters=ModelParameters(alpha=0.8, beta=3.0),
    )
