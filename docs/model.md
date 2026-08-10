# Model and implementation

This document follows one goal through the implementation. It first explains a
flat linearly solvable Markov decision process (LMDP), then shows how the
two-layer multitask LMDP (MLMDP) builds reusable physical tasks and controls
them with a smaller abstract LMDP.

## Reading guide and notation

Let:

- `n` be the number of free physical states;
- `k` be the number of subgoals;
- `g` be the selected physical goal; and
- `m = n - 1` be the number of non-goal, or **interior**, physical states.

All transition matrices use the column convention

```text
P[next_state, current_state] = probability of next_state given current_state.
```

Consequently, valid probability columns sum to one. This convention explains
the transposes in the LMDP solve and should be checked first when reproducing a
calculation outside this package.

The main symbols are:

| Symbol | Shape | Meaning |
| --- | ---: | --- |
| `P` | `n x n` | Passive physical maze dynamics |
| `D` | `n x k` | Peak-normalized subgoal profiles |
| `D_hat` | `n x k` | Profiles used for subgoal access, possibly core-gated |
| `P_i^1` | `m x m` | Layer-1 transitions that remain in the physical interior |
| `P_t^1` | `k x m` | Layer-1 transitions into subgoal boundary copies |
| `P_g^1` | `1 x m` | Layer-1 transitions into the physical goal boundary |
| `F` | `m x m` | Fundamental matrix `(I - P_i^1)^-1` |
| `H` | `(k + 1) x m` | First-hit probabilities for subgoals and the goal |
| `P_i^2` | `k x k` | Passive upper transitions between subgoals |
| `P_g^2` | `1 x k` | Passive upper termination at the physical goal |
| `Q_b` | `(k + 1) x (k + 1)` | Boundary desirabilities for reusable Layer-1 tasks |
| `Z_i` | `m x (k + 1)` | Solved interior desirabilities for those tasks |

Superscripts name the layer, not a matrix power.

## 1. Physical passive dynamics

`Maze` assigns row-major state IDs to free cells. ASCII mazes use implicit
cardinal adjacency. Edge-list mazes retain grid coordinates but explicitly
restrict which adjacent states are connected.

`LMDPEnvironment` calls
[`build_passive_dynamics`](../src/andrew_mlmdp/lmdp.py#L475) once and stores the
result as `environment.passive`.

Two passive models are available:

- `five_commands` samples north, south, east, west, or stay uniformly. Invalid
  commands remain at the current state.
- `valid_neighbors` samples uniformly from traversable cardinal neighbors and
  has no self-transition.

This matrix represents uncontrolled behavior and encodes which controlled
transitions are possible. LMDP control reweights its existing probability
mass; it never creates an impossible maze edge.

## 2. Flat first-exit LMDP

An LMDP chooses a controlled next-state distribution while paying a
KL-divergence cost for departing from the passive distribution. `lambda`
controls that trade-off: smaller values make reward more dominant, while
larger values keep the policy closer to passive motion.

The exponential transformation

```math
q(s) = \exp(r(s) / \lambda),
\qquad
z(s) = \exp(V(s) / \lambda)
```

turns the Bellman equation into a linear first-exit problem. Split the passive
dynamics into interior rows `P_II` and terminal boundary rows `P_BI`. Given
boundary desirability `z_b`, the interior solution is

```math
(I - \operatorname{diag}(q_i) P_{II}^{T}) z_i
= \operatorname{diag}(q_i) P_{BI}^{T} z_b.
```

[`solve_first_exit`](../src/andrew_mlmdp/lmdp.py#L348) performs this solve.
[`LMDPEnvironment.solve_flat`](../src/andrew_mlmdp/lmdp.py#L289) removes the
goal from the interior, uses its exponentiated terminal reward as `z_b`, and
places the solved values back into a length-`n` physical vector.

The closed-form controlled dynamics are

```math
u^*(s' \mid s)
= \frac{P(s' \mid s) z(s')}
       {\sum_y P(y \mid s) z(y)}.
```

This is implemented by
[`controlled_from_desirability`](../src/andrew_mlmdp/lmdp.py#L446). A state is
preferred when it is both reachable under `P` and desirable under `z`.

For a physical component disconnected from the goal, desirability is zero.
The high-level flat solver retains passive dynamics in an otherwise undefined
zero-mass policy column.

## 3. One representation for point and distributed subgoals

[`SubgoalBasis`](../src/andrew_mlmdp/hierarchy.py#L26) always stores a
state-by-subgoal profile matrix `D`:

- `SubgoalBasis.from_locations` creates one-hot columns.
- `SubgoalBasis.from_profiles` peak-normalizes non-negative distributed
  profiles.

Distributed profiles also have a separate immutable execution view `D_hat`.
For core threshold `tau` and exponent `gamma`, access is

```math
\widehat D_{sj}
= \left[\max\left(0,\frac{D_{sj}-\tau}{1-\tau}\right)\right]^\gamma.
```

`D` describes the learned or supplied representation. `D_hat` determines
where the rollout is allowed to enter an abstract subgoal boundary. Keeping
them separate means execution gating does not overwrite the discovered NMF
factors.

Point and distributed bases then follow the same hierarchy construction. The
only rollout-specific distinction is location semantics: accessing a point
subgoal places the current coordinate at that point; accessing a distributed
subgoal leaves the current physical coordinate unchanged.

## 4. Build the goal-conditioned Layer-1 first-exit process

`environment.hierarchy(basis, ...)` returns a reusable
[`HierarchyTemplate`](../src/andrew_mlmdp/hierarchy.py#L138).
`template.for_goal(g)` builds and caches one
[`HierarchyTask`](../src/andrew_mlmdp/hierarchy.py#L213).

For that goal, [`_build_hierarchy_task`](../src/andrew_mlmdp/hierarchy.py#L424)
does the following:

1. Remove `g` from the physical interior, leaving `m` states.
2. Create raw subgoal-access rows
   `A = alpha * D_hat[interior, :].T`, with shape `k x m`.
3. Extract physical interior transitions `P_i^1` and the physical-goal row
   `P_g^1` from `P`.
4. Stack the physical, subgoal-access, and goal transitions and renormalize
   every current-state column.

The resulting first-exit dynamics are

```text
interior rows:  P_i^1                  shape m x m
boundary rows: [P_t^1 ; P_g^1]         shape (k + 1) x m
```

where `P_t^1` is the normalized version of the access rows. A subgoal is an
additional absorbing **copy** reached probabilistically from physical states;
the original physical states remain in the maze interior. `alpha` controls the
strength of this passive access relative to ordinary movement.

The implementation is
[`_build_lower_dynamics_from_access`](../src/andrew_mlmdp/hierarchy.py#L1312).
The stacked matrix is column-stochastic after normalization.

## 5. Convert physical first hits into upper passive dynamics

The fundamental matrix

```math
F = (I - P_i^1)^{-1}
```

sums all possible amounts of time spent moving through the Layer-1 interior
before a boundary is reached. Therefore

```math
H = [P_t^1; P_g^1] F
```

gives the probability of first reaching each subgoal copy or the physical goal
from every physical interior state. It is exposed as
`task.first_hit_probabilities`.

To start an abstract transition at subgoal `j`, the construction uses the
transpose of the corresponding access row as a physical source weighting. This
gives

```math
P_i^2 = P_t^1 F P_t^{1T},
\qquad
P_g^2 = P_g^1 F P_t^{1T}.
```

After column normalization, `P_i^2` describes passive transitions among the
`k` subgoals and `P_g^2` describes passive termination at the one physical
goal boundary. See
[`_build_upper_dynamics`](../src/andrew_mlmdp/hierarchy.py#L1357).

This is the bridge between layers: long physical paths through the maze are
summarized as one small abstract transition matrix.

## 6. Solve the upper LMDP

The upper layer is another first-exit LMDP:

- its `k` interior states are subgoals;
- its one boundary state is the physical goal; and
- it uses `upper_control_cost` rather than `lower_control_cost`.

[`_solve_upper_layer`](../src/andrew_mlmdp/hierarchy.py#L1372) solves for
`task.upper_desirability` and then reweights the passive upper dynamics to
obtain `task.upper_controlled`.

At a physical state `s` that is not an entered point subgoal, the passive
abstract prediction is the first-hit column

```math
p^2(\cdot \mid s) = H[:, s].
```

The corresponding controlled prediction is obtained by multiplying this
column by upper desirability and normalizing. At an entered subgoal, the code
uses the matching columns of `upper_dynamics.passive` and
`upper_controlled` directly. See
[`compute_hierarchy_plan`](../src/andrew_mlmdp/hierarchy.py#L475).

## 7. Pre-solve the reusable lower task basis

Layer 1 needs a way to realize many abstract commands without solving a new
physical LMDP every time. It therefore pre-solves `k + 1` component tasks.

[`_build_task_basis`](../src/andrew_mlmdp/hierarchy.py#L1397) constructs the
boundary matrix `Q_b`:

- subgoal task `j` rewards subgoal boundary `j` and assigns the configured
  off-target reward to other subgoal boundaries;
- the physical-goal row is zero in every subgoal task; and
- the final component rewards only the physical goal.

Every column of `Q_b` is solved through the same Layer-1 first-exit dynamics,
producing `Z_i`. Together these are exposed as `task.task_basis`:

```text
Q_b: boundary desirabilities             (k + 1) x (k + 1)
Z_i: corresponding interior solutions    m x (k + 1)
```

Because the first-exit equation is linear in boundary desirability, a weighted
combination of these columns is also a valid desirability solution.

## 8. Turn the current upper policy into one physical policy

[`HierarchyTask.plan`](../src/andrew_mlmdp/hierarchy.py#L260) delegates to
[`_plan_from_abstract_dynamics`](../src/andrew_mlmdp/hierarchy.py#L540).
Planning performs four transformations.

First, reward inpainting converts the change requested by abstract control
into Layer-1 subgoal-boundary rewards:

```math
r^1_t = \beta\,(u^2_t - p^2_t).
```

The physical-goal boundary keeps `goal_reward`. Exponentiating these rewards
with `lower_control_cost` produces a target boundary desirability `z_target`.

Second, the target is approximated with the reusable task basis:

```math
w_{raw} = Q_b^+ z_{target},
\qquad
w = \max(0, w_{raw}).
```

`Q_b^+` is the pseudoinverse. Clipping makes the mixture non-negative. If
`include_goal_component_while_active=False`, the last weight is forced to zero
until upper termination; a best-single-subgoal fallback handles the rare case
where clipping otherwise removes every active component.

Third, the lower desirability is composed linearly:

```math
z_i^1 = Z_i w,
\qquad
z_b^1 = Q_b w.
```

Finally, the standard LMDP reweighting formula converts this desirability into
`plan.layer_one_controlled`. The complete implementation is in
[`_compose_lower_policy`](../src/andrew_mlmdp/hierarchy.py#L674).

The most useful debugging fields on `LayerOnePlan` are
`passive_abstract`, `controlled_abstract`, `inpainted_rewards`, `raw_weights`,
`weights`, `physical_desirability`, and `layer_one_controlled`.

## 9. Execute the coupled process

[`_run_hierarchical_rollout`](../src/andrew_mlmdp/hierarchy.py#L822) uses the
composed Layer-1 policy as one distribution over:

- another physical interior state;
- one of the `k` subgoal boundary copies; or
- the physical goal boundary.

The event loop is:

```text
build a plan at the start
while the physical-step limit is not reached:
    sample the current Layer-1 controlled column
    if it is a physical state:
        move, advance physical time, and optionally update online goal z
    if it is the physical goal boundary:
        move to the goal and finish
    if it is a subgoal boundary:
        record a zero-time abstract access
        sample upper termination for that entered subgoal
        if terminated: permanently install the goal-only Layer-1 plan
        otherwise: build a new plan from that upper state
        suppress another subgoal access until one physical move occurs
```

The one-physical-step refractory period prevents infinite zero-time access
loops. `Rollout.events` records the state machine explicitly using
`initial_plan`, `physical_step`, `lower_access`, `upper_command`,
`upper_termination`, and `terminal` events.

## 10. Exact versus online goal desirability

Exact execution uses the pre-solved final goal column in `Z_i`.

With `goal_learning="online"`, that column is replaced by a learned vector,
initialized to zero or supplied through `initial_goal_desirability`. Each
nonterminal physical move applies the requested number of fixed-point sweeps:

```math
z_i \leftarrow q_i(P_{II}^{T}z_i + P_{BI}^{T}z_b).
```

[`z_iteration_step`](../src/andrew_mlmdp/lmdp.py#L391) performs one sweep. The
fixed subgoal columns remain exact; only the physical-goal component is
learned. The final vector is available as
`rollout.final_goal_desirability` and can initialize the next episode.

## 11. Optional NMF discovery of distributed subgoals

NMF discovery is upstream of hierarchy execution and uses separate parameters:

1. Solve a family of flat goals with one shared environment.
2. Stack their desirabilities as `Z_tasks` with shape `n x number_of_tasks`.
3. Fit non-negative factors

   ```math
   Z_{tasks} \approx D W
   ```

   using KL-NMF.
4. Peak-normalize each column of `D` and absorb its scale into the matching row
   of `W`, preserving the reconstruction.
5. Pass `D` to `SubgoalBasis.from_profiles`, optionally applying the execution
   core gate described above.

[`discover_soft_subgoals`](../src/andrew_mlmdp/discovery.py#L201) fits every
requested rank exactly once. `study.result(k)` retrieves a cached fit rather
than rerunning NMF.

## End-to-end implementation map

| Stage | Public result | Implementation |
| --- | --- | --- |
| Physical passive dynamics | `environment.passive` | `lmdp.build_passive_dynamics` |
| Flat goal solution | `FlatSolution` | `LMDPEnvironment.solve_flat` |
| Reusable subgoal representation | `SubgoalBasis` | `SubgoalBasis.from_locations` / `from_profiles` |
| Goal-conditioned construction | `HierarchyTask` | `HierarchyTemplate.for_goal` -> `_build_hierarchy_task` |
| First-hit abstraction | `first_hit_probabilities` | `_fundamental_matrix`, `_build_upper_dynamics` |
| Upper LMDP | `upper_desirability`, `upper_controlled` | `_solve_upper_layer` |
| Reusable lower solutions | `task_basis` | `_build_task_basis` |
| Top-down composition | `LayerOnePlan` | `compute_hierarchy_plan` -> `_plan_from_abstract_dynamics` |
| Coupled execution | `Rollout` | `_run_hierarchical_rollout` |
| Distributed discovery | `NMFStudy` | `discover_soft_subgoals` |

No stage assumes a particular maze size or fixed number of subgoals. Array
shapes are derived from the supplied maze, goal partition, and profile count.
