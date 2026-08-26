def test_hierarchy_names_are_reexported():
    import andrew_mlmdp
    import andrew_mlmdp.hierarchy as hierarchy

    assert hierarchy.Task is andrew_mlmdp.Task
    assert hierarchy.Template is andrew_mlmdp.Template
    assert hierarchy.Rollout is andrew_mlmdp.Rollout
    assert hierarchy.SubgoalBasis is andrew_mlmdp.SubgoalBasis
    assert hierarchy.PairEntropy is andrew_mlmdp.PairEntropy


def test_profile_normalization_type_is_reexported():
    import andrew_mlmdp
    from andrew_mlmdp.profiles import ProfileNormalization

    assert andrew_mlmdp.ProfileNormalization is ProfileNormalization


def test_plotting_names_are_reexported():
    import andrew_mlmdp.plotting as direct_plotting
    from andrew_mlmdp import plotting

    assert plotting is direct_plotting
    assert plotting.plot_maze is direct_plotting.plot_maze
    assert (
        plotting.explore_rollout
        is direct_plotting.explore_rollout
    )


def test_connectivity_discovery_types_are_reexported():
    import andrew_mlmdp
    import andrew_mlmdp.discovery as discovery

    assert andrew_mlmdp.NMFConnectivityConfig is discovery.NMFConnectivityConfig
    assert andrew_mlmdp.NMFRestartResult is discovery.NMFRestartResult
    assert andrew_mlmdp.NMFRankResult is discovery.NMFRankResult
