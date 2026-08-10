def test_public_hierarchy_imports_remain_compatible():
    import andrew_mlmdp
    import andrew_mlmdp.hierarchy as hierarchy

    assert hierarchy.HierarchyTask is andrew_mlmdp.HierarchyTask
    assert hierarchy.HierarchyTemplate is andrew_mlmdp.HierarchyTemplate
    assert hierarchy.Rollout is andrew_mlmdp.Rollout
    assert hierarchy.SubgoalBasis is andrew_mlmdp.SubgoalBasis


def test_public_plotting_imports_remain_compatible():
    import andrew_mlmdp.plotting as direct_plotting
    from andrew_mlmdp import plotting

    assert plotting is direct_plotting
    assert plotting.plot_maze is direct_plotting.plot_maze
    assert (
        plotting.plot_interactive_soft_hierarchical_rollout
        is direct_plotting.plot_interactive_soft_hierarchical_rollout
    )
