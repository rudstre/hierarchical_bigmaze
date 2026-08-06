"""Plot one trial's navigation trajectory through the maze."""

import argparse
import os
import sys
from pathlib import Path


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

# The existing loaders resolve experiment_info relative to the data directory.
output_dir = Path.cwd()
gridmaze_code = gridmaze_root / "code"
os.chdir(gridmaze_code)
sys.path.insert(0, str(gridmaze_code))

import matplotlib.pyplot as plt
import networkx as nx

from GridMaze.core.get_sessions import MazeSession


session = MazeSession(
    subject,
    session_name,
    with_data=["trajectories_df", "trial_info_df"],
    verbose=False,
)
trial_info = session.trial_info_df
mask = (trial_info["trial"] == args.trial_id) & (trial_info["trial_phase"] == "navigation")
trajectory = session.trajectories_df.loc[mask]
if trajectory.empty:
    parser.error(f"trial {args.trial_id} has no navigation trajectory")

x = trajectory[("centroid_position", "x")]
y = trajectory[("centroid_position", "y")]
goal = trial_info.loc[mask, "goal"].iloc[0]
maze = session.simple_maze()

fig, ax = plt.subplots(figsize=(5, 5))
nx.draw_networkx(
    maze,
    pos=nx.get_node_attributes(maze, "position"),
    node_size=20,
    node_color="0.7",
    edge_color="0.7",
    with_labels=False,
    ax=ax,
)
ax.plot(x, y, color="black")
ax.scatter(x.iloc[0], y.iloc[0], color="green", label="start", zorder=3)
ax.scatter(x.iloc[-1], y.iloc[-1], color="red", label=f"goal ({goal})", zorder=3)
ax.set_aspect("equal")
ax.axis("off")
ax.legend()

output = output_dir / f"{subject}_{session_name}_trial_{args.trial_id}.png"
fig.savefig(output, bbox_inches="tight", dpi=150)
print(output)
