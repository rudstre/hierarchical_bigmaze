# Model and implementation

This note defines the mathematical conventions used by the package and maps
the paper's construction to the corresponding code. It is intended to be read
beside `lmdp.py` and `hierarchy.py`.

## 1. State and matrix conventions

An LMDP state space is divided into interior states `I` and first-exit boundary
states `B`. The package stores passive dynamics as

```text
P[next_state, current_state] = P(next_state | current_state).
```

Columns therefore sum to one. `FirstExitDynamics` holds the two row blocks:

| Field | Shape | Meaning |
| --- | --- | --- |
| `interior_passive` | `(n_i, n_i)` | `P_II`, transitions into interior states |
| `boundary_passive` | `(n_b, n_i)` | `P_BI`, transitions into boundary states |
| `passive` | `(n_i + n_b, n_i)` | the vertical stack of both blocks |

Rows and columns are deliberately not hidden behind an automatic orientation
conversion. This makes every matrix multiplication directly inspectable.

## 2. First-exit LMDP

For state reward `r`, a layer's control cost `lambda`, value `V`, and
desirability `z`,

```math
q(s) = \exp(r(s)/\lambda), \qquad
z(s) = \exp(V(s)/\lambda).
```

The project uses a uniform interior reward, although `solve_first_exit` also
accepts one exponentiated reward per interior state. Boundary desirability is
fixed by the terminal task. Flat and lower-layer calculations use
`lower_control_cost`; the abstract calculation uses `upper_control_cost`.
Tasks composed within a layer always share that layer's control cost.

With this package's column-stochastic convention, paper Equation 4 is

```math
(I - \operatorname{diag}(q_i)P_{II}^{T})z_i
= \operatorname{diag}(q_i)P_{BI}^{T}z_b.
```

`solve_first_exit` constructs this equation literally and uses
`numpy.linalg.solve`. `solve_desirability` is the maze convenience wrapper: it
chooses one physical goal as the boundary, constructs the two passive blocks,
and places the resulting `z_i` back into `maze.free_cells` order.

Paper Equation 6 gives the controlled next-state distribution:

```math
a^*(s'|s) =
\frac{P(s'|s)z(s')}{\sum_y P(y|s)z(y)}.
```

`controlled_from_desirability` implements this for any rectangular first-exit
matrix. `controlled_dynamics` applies it to the square physical random walk.
The outgoing column of an absorbing goal is not used: rollouts terminate as
soon as the goal is entered.

## 3. Physical passive dynamics

`build_passive_dynamics` assigns equal probability to five commands: north,
south, east, west, and stay. A command into a wall or outside the grid returns
the agent to its current cell. Several invalid commands can therefore add mass
to the same self-transition. Maze geometry itself knows nothing about goals or
subgoals.

## 4. Lower-layer augmentation

For one selected physical goal, that goal is removed from the lower interior
set and becomes the original terminal boundary. Every configured subgoal keeps
its ordinary physical state and receives a separate abstract boundary copy.

The access matrix `P_t` places mass `alpha` only from a subgoal's physical cell
to its own copy. The augmented passive matrix is

```math
\widetilde P^1 = \mathcal N([P_i^1; P_t^1; P_g^1]),
```

where `N` normalizes each column. In `TwoLayerModel`:

| Quantity | Code |
| --- | --- |
| `P_i^1` | `model.lower_dynamics.interior_passive` |
| `[P_t^1; P_g^1]` | `model.lower_dynamics.boundary_passive` |
| `P_t^1` | `model.lower_subgoal_passive` |
| `P_g^1` | `model.lower_goal_passive` |

The boundary order is always the supplied subgoal order followed by the goal.

### Distributed soft subtasks

The discovery extension first solves a goal-task ensemble and collects the
full physical desirabilities as columns of `Z`. KL-NMF gives

```math
Z \approx D W, \qquad D,W\geq 0.
```

The ensemble uses an immutable `NMFDiscoveryParameters` object containing only
its interior reward, goal reward, and control cost. These are not execution
parameters. Once `D` is learned, the same profiles are reused while
`ModelParameters` controls hierarchy execution. In particular, changing
execution `lower_control_cost` does not require or trigger NMF rediscovery.
Passing a legacy `ModelParameters` object to `build_goal_task_ensemble`
remains supported, but the three discovery fields are copied into a separate
`NMFDiscoveryParameters` object.

For an ensemble in which every task uses the same terminal reward, discovery
has only one profile-shaping degree of freedom. The task solutions depend on
reward/control-cost ratios; a shared terminal reward additionally multiplies
every column of `Z` by the same scalar. After the component-wise gauge below,
the ideal profile geometry therefore depends only on
`-interior_reward / control_cost`. The sweep fixes the other values at a
numerically safe scale and searches this ratio broadly.

The execution equations have a related common gauge: multiplying
`interior_reward`, `goal_reward`, both control costs, `off_target_reward`, and
`beta` by one positive constant leaves every exponentiated reward ratio
unchanged. `alpha` is excluded because it is literal passive access mass. The
defaults jointly use a narrow 80% execution core, `alpha=0.2`, and
`beta=13.0`. The core removes incidental fringe accesses, while the increased
access strength preserves deliberate upper-level transitions on the reduced
support.

`factorize_soft_subtasks` factorizes the actual `Z` and fixes the independent
gauge of every NMF component,

```math
c_j=\max_s D_{sj},\qquad D_{:j}\leftarrow D_{:j}/c_j,\quad
W_{j:}\leftarrow c_jW_{j:}.
```

This leaves `D @ W` unchanged and makes every profile peak at one.
`display_profiles` returns a separately allocated plotting view with the same
peak convention.
By default, `build_soft_two_layer_model` restricts access to a soft profile
core before replacing the one-hot access rows:

```math
\widehat D_{sj}
=
\left[
\max\left(0,\frac{D_{sj}-\tau}{1-\tau}\right)
\right]^\gamma,
\qquad
P_t = \alpha \widehat D^T,
```

where every input column is first scaled by its own peak, and the defaults are
`tau = 0.8` and `gamma = 1`. The complete augmented passive columns are then
normalized. Because the transformed profiles still peak at one, `alpha`
remains the maximum local passive access strength; controlled access can
differ after desirability reweighting. Pass `core_threshold=None` to use
custom profiles exactly as supplied and recover direct paper Equation 3
access. The physical-goal row, first-hit abstraction, upper solve, task basis,
and reward inpainting are all rebuilt from the same transformed access
profiles.

The 80% threshold is an execution choice, not an NMF discovery constraint.
It excludes weak profile fringes that can otherwise fire while the agent is
merely crossing a doorway en route to a different commanded region. The
discovery sweep continues to measure broader 25% profile support separately;
changing execution gating does not refactorize or otherwise alter `D`.

After a lower access into upper state \(j\), the entered column programs the
next lower plan:

```math
r_t^1 \propto a_i^2(\cdot\mid j)-p_i^2(\cdot\mid j).
```

Because this implementation has exactly two layers, it samples only the
terminal versus nonterminal upper outcome. A terminal outcome permanently
selects the physical-goal-only task without teleporting. All subtask rows are
refractory until one physical transition has occurred, preventing zero-time
cycles between overlapping distributed profiles.

## 5. First-hit abstraction and upper layer

Let

```math
F = (I - \widetilde P_i^1)^{-1}.
```

The implementation obtains `F` with a linear solve against the identity. The
probability of first exiting through each lower boundary from each lower
interior state is

```math
H = [\widetilde P_t^1; \widetilde P_g^1]F.
```

This is `model.first_hit_probabilities`. Paper Equations 8 and 9 then define
the upper passive blocks:

```math
P_{II}^2 = \widetilde P_t^1 F \widetilde P_t^{1T},
\qquad
P_{BI}^2 = \widetilde P_g^1 F \widetilde P_t^{1T}.
```

They are stored in `model.upper_dynamics`. Upper columns follow subgoal order;
the single upper boundary row is the selected physical goal. The upper layer
uses the same interior reward and goal reward as the lower layer, but
exponentiates them with `upper_control_cost`.

`build_subgoal_passive_dynamics` performs the task-independent form of
Equation 8 before adding a selected physical goal. This is the square graph
drawn over the maze in the Figure 3a-style plot.

## 6. Multitask basis

`TaskBasis` stores the paper's two multitask matrices:

| Field | Symbol | Shape |
| --- | --- | --- |
| `boundary_desirability` | `Q_b` | `(n_boundary, n_tasks)` |
| `interior_desirability` | `Z_i` | `(n_interior, n_tasks)` |

Each subgoal component task has high boundary reward at one subgoal copy and
the configured off-target reward at the others. The physical-goal component is
kept in a separate block, following the augmented basis in the paper. Each
column of `Z_i` is solved once with Equation 4.

For desired boundary desirability `q`, paper Equation 7 is approximated as

```math
w_{raw} = Q_b^\dagger q, \qquad w = \max(0, w_{raw}),
```

and linearity gives

```math
z_i = Z_i w.
```

`LayerOnePlan` retains the desired boundary vector, raw and clipped weights,
reconstructed boundary vector, physical desirability, and controlled policy so
projection error and task composition can be inspected.

## 7. Reward inpainting

At an upper interior state, Equation 10 communicates the difference between
controlled and passive upper dynamics:

```math
r_t^1 = \beta(a_i^2(\cdot|s) - P_i^2(\cdot|s)).
```

The paper specifies proportionality but not the scale. The historical
eight-component soft-hierarchy default uses `beta=3`. The signal applies to
subgoal-copy rewards; the original physical
goal retains its terminal reward. `compute_layer_one_plan` exponentiates the
result with `lower_control_cost`, projects it through `Q_b`, and composes the
lower desirability.

For a physical start that is not a subgoal, the implementation uses the
corresponding column of `first_hit_probabilities` as a temporary upper passive
distribution. This avoids adding a persistent start node and is an explicit
project convention.

## 8. Hierarchical execution

`sample_hierarchical_rollout` repeats this cycle:

1. Sample a row of the active lower controlled distribution.
2. If it is a physical interior row, advance physical time by one step.
3. If it is the physical goal boundary, advance the final step and terminate.
4. If it is a subgoal copy, consume no physical time, recompute the task blend
   from that abstract state, and continue from its physical cell.

Immediately re-accessing the subgoal that supplied the active plan changes no
state, reward, or weight. The sampler removes that outcome and renormalizes,
analytically marginalizing repeated zero-time no-ops. Both physical-step and
abstract-access limits return explicit status values.

For online goal learning, `sample_online_hierarchical_rollout` and
`sample_online_soft_hierarchical_rollout` replace the exact physical-goal
interior basis column with a learned vector. It starts at zero unless an
initial vector is supplied and receives the requested number of full
Z-iteration sweeps after each nonterminal physical transition. Abstract
accesses consume no physical time and do not advance goal learning. The soft
result records distributed accesses as `SoftSubtaskAccess` values.

## 9. Reading order

For a code-level walkthrough, read in this order:

1. `maze.py`: geometry and state numbering.
2. `FirstExitDynamics`, `solve_first_exit`, and
   `controlled_from_desirability` in `lmdp.py`.
3. `build_two_layer_model` and its short private construction helpers.
4. `compute_layer_one_plan`.
5. `sample_hierarchical_rollout`.

The test suite checks the Bellman equation, Equation 6, first-hit probabilities
against Monte Carlo simulation, exact composition, inpainting, legal rollouts,
and the canonical four-room numerical outputs.
