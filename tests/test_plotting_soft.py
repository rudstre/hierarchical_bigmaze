import plotly.graph_objects as go

from andrew_mlmdp import Task, plotting


def test_soft_player_recomputes_staged_locations_once(
    soft_corridor_template, monkeypatch
):
    original_rollout_method = Task.rollout
    rollout_calls = 0

    def counted_rollout(self, *args, **kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        return original_rollout_method(self, *args, **kwargs)

    monkeypatch.setattr(Task, "rollout", counted_rollout)
    player = plotting.explore_rollout(
        soft_corridor_template, (0, 0), (1, 3), seed=2, max_steps=100
    )
    assert isinstance(player.figure, go.Figure)
    original_rollout = player.rollout
    player._location_state["pending_start"] = (0, 1)
    player._location_state["pending_goal"] = (1, 2)
    player.recompute()
    assert rollout_calls == 2
    assert player.start == (0, 1)
    assert player.goal == (1, 2)
    assert player.rollout is not original_rollout
    assert player.frame_index == 0


def test_soft_player_controls_and_heatmap(soft_corridor_template):
    player = plotting.explore_rollout(
        soft_corridor_template, (0, 0), (1, 3), seed=3, max_steps=100
    )
    player.show_goal_component(False)
    player.show_normalization(False)
    player.show_frame(player.frame_count - 1)
    assert not player.goal_component_visible
    assert not player.frame_normalization
    assert player.frame_index == player.frame_count - 1
    desirability = [trace for trace in player.figure.data if trace.type == "heatmap"][
        -1
    ]
    goal_row, goal_column = player.goal
    assert desirability.z[goal_row][goal_column] is not None
