"""Plot one trial's navigation trajectory through the maze."""

import argparse
import sys
from pathlib import Path
from typing import cast

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument("--session_id", required=True, help="e.g. m2/2022-06-23.maze")
parser.add_argument("--trial_id", required=True, type=int)
args = parser.parse_args()

try:
    subject, session_name = args.session_id.split("/")
except ValueError:
    parser.error("SESSION_ID must look like m2/2022-06-23.maze")

project_root = Path(__file__).resolve().parents[1]
gridmaze_root = project_root / "external" / "GridMaze-mFC-ephys-DATA"
processed_data_root = gridmaze_root / "data" / "processed_data"
session_dir = processed_data_root / subject / session_name
if not session_dir.is_dir():
    parser.error(f"session not found: {session_dir}")

output_dir = Path.cwd()
gridmaze_code = gridmaze_root / "code"
sys.path.insert(0, str(gridmaze_code))

# GridMaze is an external checkout rather than an installed package.
from GridMaze.core.get_sessions import MazeSession  # noqa: E402

session = MazeSession(
    subject,
    session_name,
    with_data=["trajectories_df", "trial_info_df"],
    verbose=False,
)
trial_info = session.trial_info_df
trajectories = session.trajectories_df
if trial_info is None or trajectories is None:
    parser.error("required trajectory data is missing")
mask = (trial_info["trial"] == args.trial_id) & (
    trial_info["trial_phase"] == "navigation"
)
trajectory = trajectories.loc[mask]
if trajectory.empty:
    parser.error(f"trial {args.trial_id} has no navigation trajectory")

x = trajectory[("centroid_position", "x")]
y = trajectory[("centroid_position", "y")]
goal_series = cast(pd.Series, trial_info.loc[mask, "goal"])
goal = goal_series.iloc[0]
maze = session.simple_maze()

positions = nx.get_node_attributes(maze, "position")
edge_x = []
edge_y = []
for source, destination in maze.edges:
    source_x, source_y = positions[source]
    destination_x, destination_y = positions[destination]
    edge_x.extend([source_x, destination_x, None])
    edge_y.extend([source_y, destination_y, None])

figure = go.Figure()
figure.add_trace(
    go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"color": "#b3b3b3", "width": 1},
        hoverinfo="skip",
        showlegend=False,
    )
)
figure.add_trace(
    go.Scatter(
        x=[positions[node][0] for node in maze.nodes],
        y=[positions[node][1] for node in maze.nodes],
        mode="markers",
        marker={"size": 5, "color": "#b3b3b3"},
        hoverinfo="skip",
        showlegend=False,
    )
)
figure.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": "black", "width": 2},
        name="trajectory",
    )
)
figure.add_trace(
    go.Scatter(
        x=[x.iloc[0]],
        y=[y.iloc[0]],
        mode="markers",
        marker={"size": 10, "color": "green"},
        name="start",
    )
)
figure.add_trace(
    go.Scatter(
        x=[x.iloc[-1]],
        y=[y.iloc[-1]],
        mode="markers",
        marker={"size": 11, "color": "red"},
        name=f"goal ({goal})",
    )
)
figure.update_layout(
    width=500,
    height=500,
    template="plotly_white",
    xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
    yaxis={"visible": False},
)
output = output_dir / f"{subject}_{session_name}_trial_{args.trial_id}.png"
figure.write_image(output, width=750, height=750)
print(output)
