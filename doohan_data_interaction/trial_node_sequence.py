"""Print the maze nodes entered during one trial's navigation phase."""

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

gridmaze_code = gridmaze_root / "code"
os.chdir(gridmaze_code)
sys.path.insert(0, str(gridmaze_code))

from GridMaze.core.get_sessions import MazeSession


session = MazeSession(
    subject,
    session_name,
    with_data=["trajectories_df", "trial_info_df"],
    verbose=False,
)
trial_info = session.trial_info_df
mask = (trial_info["trial"] == args.trial_id) & (trial_info["trial_phase"] == "navigation")
positions = session.trajectories_df.loc[mask, ("maze_position", "simple")].dropna()
if positions.empty:
    parser.error(f"trial {args.trial_id} has no navigation trajectory")

entered = positions[positions.ne(positions.shift())]
nodes = [position for position in entered if "-" not in position]
print(nodes)
