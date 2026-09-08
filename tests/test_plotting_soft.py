import importlib

from plotly.basedatatypes import BaseFigure

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
    assert isinstance(player.figure, BaseFigure)
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


def test_soft_player_figure_is_a_live_widget(soft_corridor_template):
    """A static figure would render once and ignore every control callback."""

    widgets = importlib.import_module("ipywidgets")
    player = plotting.explore_rollout(
        soft_corridor_template, (0, 0), (1, 3), seed=3, max_steps=100
    )
    assert isinstance(player.figure, widgets.Widget)
    assert isinstance(player.panel, widgets.Widget)
    assert list(player.panel.children) == [player.controls, player.figure]


def test_soft_player_frame_controls_redraw_the_figure(soft_corridor_template):
    player = plotting.explore_rollout(
        soft_corridor_template, (0, 0), (1, 3), seed=3, max_steps=100
    )
    previous_button, next_button = player.controls.children[0].children[:2]
    assert previous_button.disabled
    assert not next_button.disabled
    first_frame_text = player.figure.data[-1].text
    next_button.click()
    assert player.frame_index == 1
    assert not previous_button.disabled
    assert player.figure.data[-1].text != first_frame_text
    player.show_frame(player.frame_count - 1)
    assert next_button.disabled


def test_soft_player_traces_the_requested_goal_learning(soft_corridor_template):
    player = plotting.explore_rollout(
        soft_corridor_template,
        (0, 0),
        (1, 3),
        goal_learning="online",
        seed=3,
        max_steps=100,
    )
    assert player.rollout.goal_learning == "online"
    player._location_state["pending_start"] = (0, 1)
    player._location_state["pending_goal"] = (1, 2)
    player.recompute()
    assert player.rollout.goal_learning == "online"
