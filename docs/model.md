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
[`build_passive_dynamics`](../src/andrew_mlmdp/lmdp.py) once and stores the
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

[`solve_first_exit`](../src/andrew_mlmdp/lmdp.py) performs this solve.
[`LMDPEnvironment.solve_flat`](../src/andrew_mlmdp/lmdp.py) removes the
goal from the interior, uses its exponentiated terminal reward as `z_b`, and
places the solved values back into a length-`n` physical vector.

The closed-form controlled dynamics are

```math
u^*(s' \mid s)
= \frac{P(s' \mid s) z(s')}
       {\sum_y P(y \mid s) z(y)}.
```

This is implemented by
[`controlled_from_desirability`](../src/andrew_mlmdp/lmdp.py). A state is
preferred when it is both reachable under `P` and desirable under `z`.

For a physical component disconnected from the goal, desirability is zero.
The high-level flat solver retains passive dynamics in an otherwise undefined
zero-mass policy column.

## 3. One representation for point and distributed subgoals

[`SubgoalBasis`](../src/andrew_mlmdp/hierarchy/core.py) always stores a
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
[`HierarchyTemplate`](../src/andrew_mlmdp/hierarchy/core.py).
`template.for_goal(g)` builds and caches one
[`HierarchyTask`](../src/andrew_mlmdp/hierarchy/core.py).

For that goal, [`_build_hierarchy_task`](../src/andrew_mlmdp/hierarchy/core.py)
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
[`_build_lower_dynamics_from_access`](../src/andrew_mlmdp/hierarchy/core.py).
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
[`_build_upper_dynamics`](../src/andrew_mlmdp/hierarchy/core.py).

This is the bridge between layers: long physical paths through the maze are
summarized as one small abstract transition matrix.

## 6. Solve the upper LMDP

The upper layer is another first-exit LMDP:

- its `k` interior states are subgoals;
- its one boundary state is the physical goal; and
- it uses `upper_control_cost` rather than `lower_control_cost`.

[`_solve_upper_layer`](../src/andrew_mlmdp/hierarchy/core.py) solves for
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
[`compute_hierarchy_plan`](../src/andrew_mlmdp/hierarchy/core.py).

## 7. Pre-solve the reusable lower task basis

Layer 1 needs a way to realize many abstract commands without solving a new
physical LMDP every time. It therefore pre-solves `k + 1` component tasks.

[`_build_task_basis`](../src/andrew_mlmdp/hierarchy/core.py) constructs the
interior basis `Z_i` from the fixed boundary matrix `Q_b` stored by
`LayerOneTaskLibrary`. The matrix is not constructed from behavioral rewards
or control costs. The standard canonical library has:

- target subgoal desirability `1`;
- off-target subgoal desirability `exp(-18)`;
- the physical-goal row is zero in every subgoal task; and
- a final physical-goal component with desirability `1`.

For eight subgoals this is a full-rank `9 x 9` matrix: one common subgoal
mode, seven subgoal-contrast modes, and one physical-goal mode. Its condition
number is approximately `1.00000012184`. `from_desirabilities(...)` records
the three canonical construction values as metadata;
`LayerOneTaskLibrary.from_matrix(...)` instead accepts any validated finite,
non-negative, full-rank matrix and leaves that optional metadata unset. The
immutable matrix itself is always the source of truth.

Every column of `Q_b` is solved through the same Layer-1 first-exit dynamics,
producing `Z_i`. Together these are exposed as `task.task_basis`:

```text
Q_b: boundary desirabilities             (k + 1) x (k + 1)
Z_i: corresponding interior solutions    m x (k + 1)
```

Because the first-exit equation is linear in boundary desirability, a weighted
combination of these columns is also a valid desirability solution.

## 8. Turn the current upper policy into one physical policy

[`HierarchyTask.plan`](../src/andrew_mlmdp/hierarchy/core.py) delegates to
[`_plan_from_abstract_dynamics`](../src/andrew_mlmdp/hierarchy/core.py).
Planning performs four transformations.

First, reward inpainting converts the change requested by abstract control
into Layer-1 boundary rewards for every possible abstract outcome, including
the physical goal:

```math
r^1_t = \beta\,(u^2_t - p^2_t).
```

The active physical-goal reward is the same probability-difference term, not
the behavioral `goal_reward`. Exponentiating these rewards with
`lower_control_cost` produces a target boundary desirability `z_target`.
`goal_reward` instead remains active in upper termination and in the
behavioral goal-only policy installed after upper termination; that goal-only
policy is solved directly and does not depend on the fixed goal-library
column.

Second, the target is approximated with the reusable task basis:

```math
w_{raw} = Q_b^+ z_{target},
\qquad
w = \max(0, w_{raw}).
```

`Q_b^+` is the pseudoinverse. Clipping makes the mixture non-negative. With
`composition_exponent = c`, only the first `k` positive subgoal weights are
then redistributed:

```math
p_j = \frac{w_j}{\sum_{l<k} w_l},
\qquad
\widetilde w_j
= \frac{p_j^c}{\sum_{l<k}p_l^c}\sum_{l<k}w_l.
```

The transform preserves total subgoal weight mass and leaves the final
physical-goal weight exactly unchanged. Consequently it does not directly
change subgoal-versus-goal allocation in weight space, although the physical
policy can change because mass moves among distinct `Z_i` columns. `c = 1`
is the exact unsharpened path, `c < 1` is more diffuse, and `c > 1` is more
competitive. Fractional powers are evaluated only at strictly positive
weights, preserving exact zeros and finite inactive-entry gradients. The
diagnostic hard winner-take-all mode preserves subgoal mass, splits exact ties,
and cannot be used with Adam.

Third, the lower desirability is composed linearly:

```math
z_i^1 = Z_i w,
\qquad
z_b^1 = Q_b w.
```

Finally, the standard LMDP reweighting formula converts this desirability into
`plan.layer_one_controlled`. The complete implementation is in
[`_compose_lower_policy`](../src/andrew_mlmdp/hierarchy/core.py).

The most useful debugging fields on `LayerOnePlan` are
`passive_abstract`, `controlled_abstract`, `inpainted_rewards`, `raw_weights`,
`weights`, `physical_desirability`, and `layer_one_controlled`.

## 9. Execute the coupled process

[`_run_hierarchical_rollout`](../src/andrew_mlmdp/hierarchy/rollout.py) uses the
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

For ordinary pre-termination plans, exact execution uses the fixed final
goal-library column in `Z_i`. After upper termination, the goal-only plan is
instead solved directly from behavioral `goal_reward`.

With `goal_learning="online"`, that column is replaced by a learned vector,
initialized to zero or supplied through `initial_goal_desirability`. Each
nonterminal physical move applies the requested number of fixed-point sweeps:

```math
z_i \leftarrow q_i(P_{II}^{T}z_i + P_{BI}^{T}z_b).
```

[`z_iteration_step`](../src/andrew_mlmdp/lmdp.py) performs one sweep. The
fixed subgoal columns remain exact; only the physical-goal component is
learned. The final vector is available as
`rollout.final_goal_desirability` and can initialize the next episode.

## 11. Optional NMF discovery of distributed subgoals

NMF discovery is upstream of hierarchy execution and uses separate parameters.
It first solves a configurable family of flat goals and stacks their
desirabilities as `Z_tasks = X`, with shape `n x number_of_tasks`. Each
requested rank `k` produces non-negative factors

```math
X \approx D W,
\qquad
D \in \mathbb{R}_{\ge 0}^{n \times k},
\quad
W \in \mathbb{R}_{\ge 0}^{k \times number\_of\_tasks}.
```

With the default `lambda_smooth=0`, discovery uses the original
scikit-learn multiplicative-update KL-NMF solver and performs no graph work.

For a positive strength, binary adjacency `A` comes from supported non-self
transitions in `environment.passive`. Thus it respects maze connections and
state order without using Euclidean distance. With degree matrix `Delta` and
`L = Delta - A`, the optimized objective is

```math
J(D,W)
= KL_{raw}(X \mid\mid D W)
+ \lambda_{smooth}\,\operatorname{Tr}(D^T L D).
```

The Laplacian term applies only to `D` and equals the sum of squared profile
differences over undirected graph edges. Defining `R = X / (D W)`, one
regularized multiplicative-update sweep is

```math
W \leftarrow W \odot
\frac{D^T R}{D^T \mathbf{1}},
```

followed by recomputing `R` and applying

```math
D \leftarrow D \odot
\frac{R W^T + 2\lambda_{smooth} A D}
     {\mathbf{1}W^T + 2\lambda_{smooth}\Delta D}.
```

These updates preserve non-negativity. They are derived for the unconstrained
objective. After each regularized sweep, the implementation additionally
fixes the NMF scale gauge by enforcing `max(D[:, j]) = 1` and absorbing the
scale into row `j` of `W`. Post-normalization objective descent is therefore
verified empirically rather than assumed from the standard MU guarantee.

[`discover_soft_subgoals`](../src/andrew_mlmdp/discovery.py) fits every
requested rank once, and `study.result(k)` retrieves that cached fit.
Regularized results expose the initial full objective and one value per
iteration through `objective_history`. The existing
`reconstruction_error` remains normalized generalized KL only, with no
smoothness penalty. The returned peak-normalized `D` can then be passed to
`SubgoalBasis.from_profiles`, optionally applying the execution gate
described above.

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

## Differentiable hierarchical likelihood and fitting

The NumPy hierarchy remains the rollout and regression implementation. The
Torch likelihood rebuilds every parameter-dependent quantity in float64 for
each forward graph, while retaining the `P[next_state, current_state]`
orientation and exact latent controller-mode semantics. Only topology,
indices, physical passive dynamics, and fixed normalized subgoal profiles are
safe to retain across optimizer steps. Gated access profiles, lower and upper
dynamics, task bases, plans, policies, latent kernels, and first-departure
occupancies must be recomputed.

For full datasets, parameter-independent collapsed trajectories and integer context
indices are prepared once. Each forward graph then constructs a shared boundary
task pseudoinverse, goal-specific plan banks, dense shared controller columns,
start-specific initial columns, and all complete-mode closure systems. The closure
systems and observed departure operators are assembled by indexed tensor operations
and solved in one batched float64 call; only the final trajectory recursion remains
sequential. All differentiable banks are discarded after that graph.

Gate structure belongs to `SubgoalBasis`: point and ungated soft bases cannot
acquire gate parameters during fitting. For gated soft bases, threshold and
exponent defaults come from the basis rather than legacy fields on
`ModelParameters`. The hard gate is intentionally piecewise differentiable.
Entries below threshold have no local branch gradient, although gradients from
active entries may move the global threshold enough to activate them later. If
only exact profile peaks remain active, threshold and exponent can be weakly
identified or have zero gradients because gated peaks remain exactly one.

For a set of physical goals `G`, the complete hierarchy is structurally
defined only below

```math
\tau_{max}=\min_{g\in G,j}\max_{s\ne g}D_{sj}.
```

`HierarchyTemplate.core_threshold_domain(goals)` reports this strict bound and
all limiting `(goal, subgoal_index)` pairs. Goal-task construction and fitting
reject public initial thresholds outside the corresponding domain. During
fitting, the private raw transform maps into
`(DOMAIN_EPS, nextafter(tau_max, -inf))`; the one-ULP upper margin prevents a
saturated sigmoid from eliminating the final support state. This changes
neither the public physical threshold nor the hard-gate equation.

`fit_hierarchical_model_parameters` minimizes the negative summed trajectory
log-likelihood using a private constrained parameterization. It never mutates
`ModelParameters`, `SubgoalBasis`, `HierarchyTemplate`, or NumPy caches. Its
`best_parameter_values` snapshot can be passed explicitly to
`total_hierarchical_movement_log_likelihood_torch`. NumPy rollout from fitted
values requires separately constructing a fresh basis and template.

The fixed task library and `composition_exponent` are not Adam variables.
Adam fitting currently fixes `composition_exponent = c = 1.0` and rejects a
template configured with another value. The behavioral fit contains
`interior_reward`, `goal_reward`, `lower_control_cost`, `upper_control_cost`,
`alpha`, `beta`, and, when the basis gate is active, `core_threshold` and
`core_exponent`. The former behavioral `off_target_reward` no longer exists;
canonical task-library metadata calls its replacement
`basis_off_target_desirability` and never repurposes it as a fitted reward.

Adam uses PyTorch's `ReduceLROnPlateau` with the evaluated pre-update loss passed to
the scheduler only after its aligned optimizer update. When the scheduler lowers
the learning rate, fitting restores the globally best raw parameter state and starts
a fresh Adam optimizer and scheduler at that lower rate. This discards momentum
from the coarser stage and gives every finer stage its own plateau bookkeeping while
preserving the global best state. Each evaluation records the learning rate active
for that state; a reduction and best-state restart therefore first appear on the
following evaluation. Plateau convergence patience is inactive while another
learning-rate stage remains and starts fresh only after the minimum learning rate is
active. The default schedule starts at `0.05`, uses factor `0.3` with plateau
patience `7`, and has a minimum learning rate of `1e-5`.

Optimization plateau scales are independent of `relative_tolerance`.
`scheduler_relative_threshold` is passed to `ReduceLROnPlateau`, while
`convergence_relative_threshold` defines meaningful best-loss improvement for
patience at the minimum learning rate. Omitting either option preserves legacy
behavior by falling back to `relative_tolerance`; this keeps existing fitting
calls compatible while allowing plateau decisions to use a scientifically
interpretable likelihood scale.
