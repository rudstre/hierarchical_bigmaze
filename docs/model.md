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
| `D` | `n x k` | Subgoal profiles, peak-normalized by default or optionally L2-normalized |
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

`Environment` calls
[`passive_dynamics`](../src/andrew_mlmdp/lmdp.py) once and stores the
result as `environment.passive`.

Two passive models are available:

- `valid_neighbors` is the default. It samples uniformly from traversable
  cardinal neighbors and has no self-transition.
- `five_commands` is opt-in. It samples north, south, east, west, or stay
  uniformly; invalid commands remain at the current state.

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
[`Environment.solve`](../src/andrew_mlmdp/lmdp.py) removes the
goal from the interior, uses its exponentiated terminal reward as `z_b`, and
places the solved values back into a length-`n` physical vector.

The closed-form controlled dynamics are

```math
u^*(s' \mid s)
= \frac{P(s' \mid s) z(s')}
       {\sum_y P(y \mid s) z(y)}.
```

This is implemented by
[`controlled_dynamics`](../src/andrew_mlmdp/lmdp.py). A state is
preferred when it is both reachable under `P` and desirable under `z`.

For a physical component disconnected from the goal, desirability is zero.
The high-level flat solver retains passive dynamics in an otherwise undefined
zero-mass policy column.

## 3. One representation for point and distributed subgoals

[`SubgoalBasis`](../src/andrew_mlmdp/hierarchy/model.py) always stores a
state-by-subgoal profile matrix `D`:

- `SubgoalBasis.from_locations` creates one-hot columns.
- `SubgoalBasis.from_profiles` normalizes non-negative distributed profiles
  by their peak by default, or by their L2 norm when
  `profile_normalization="l2"`.

Distributed profiles also have a separate immutable execution view `D_hat`.
Let `D_tilde[:, j] = D[:, j] / max(D[:, j])`. For core threshold `tau`
and exponent `gamma`, the unnormalized gated access profile is

```math
G_{sj}
= \left[\max\left(0,\frac{\widetilde D_{sj}-\tau}{1-\tau}\right)\right]^\gamma,
\qquad
\widehat D_{:j}=\mathcal N_m(G_{:j}),
```

where `m` is the configured normalization mode,

```math
\mathcal N_{peak}(x)=\frac{x}{\max_s x_s},
\qquad
\mathcal N_{l2}(x)=\frac{x}{\lVert x\rVert_2}.
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
[`Template`](../src/andrew_mlmdp/hierarchy/model.py).
`template.task(g)` builds and caches one
[`Task`](../src/andrew_mlmdp/hierarchy/model.py).

For that goal, [`_build_task`](../src/andrew_mlmdp/hierarchy/model.py)
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

These three quantities must not be conflated: `basis.profiles` contains the
original normalized NMF representation, `basis.access_profiles` contains
the reusable gated profile in the same normalization mode, and
`task.subtask_access` contains `P_t^1`,
the goal-conditioned execution-access transition probabilities after the full
augmented passive matrix has been normalized. The last quantity is not `D`,
`D_hat`, or an NMF profile.

The implementation is
[`_lower_dynamics`](../src/andrew_mlmdp/hierarchy/model.py).
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
`task.first_hit`.

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
[`_upper_dynamics`](../src/andrew_mlmdp/hierarchy/model.py).

This is the bridge between layers: long physical paths through the maze are
summarized as one small abstract transition matrix.

## 6. Solve the upper LMDP

The upper layer is another first-exit LMDP:

- its `k` interior states are subgoals;
- its one boundary state is the physical goal; and
- it uses `upper_control_cost` rather than `lower_control_cost`.

[`_solve_upper`](../src/andrew_mlmdp/hierarchy/model.py) solves for
`task.upper_desirability` and then reweights the passive upper dynamics to
obtain `task.upper_controlled`.

The default reward gauge is `interior_reward = -1` and `goal_reward = 0`.
Before fixing this gauge, multiplying `interior_reward`, both control costs,
and `beta` by one positive constant left every policy unchanged. Hierarchy
factory defaults are execution choices distinct from flat `Parameters()` and
NMF discovery defaults.

At a physical state `s` that is not an entered point subgoal, the passive
abstract prediction is the first-hit column

```math
p^2(\cdot \mid s) = H[:, s].
```

The corresponding controlled prediction is obtained by multiplying this
column by upper desirability and normalizing. At an entered subgoal, the code
uses the matching columns of `upper_dynamics.passive` and
`upper_controlled` directly. See
[`compute_plan`](../src/andrew_mlmdp/hierarchy/model.py).

## 7. Pre-solve the reusable lower task basis

Layer 1 needs a way to realize many abstract commands without solving a new
physical LMDP every time. It therefore pre-solves `k + 1` component tasks.

[`_task_basis`](../src/andrew_mlmdp/hierarchy/model.py) constructs the
interior basis `Z_i` from the fixed boundary matrix `Q_b` stored by
`TaskLibrary`. The matrix is not constructed from behavioral rewards
or control costs. The standard canonical library has:

- target subgoal desirability `1`;
- off-target subgoal desirability `0`;
- the physical-goal row is zero in every subgoal task; and
- a final physical-goal component with desirability `1`.

Thus `Q_b` is the `(k + 1) x (k + 1)` identity matrix, with every singular
value and its condition number equal to `1`. The zero off-target value is the
desirability-space representation of an exclusive terminal task. It is not
derived from `interior_reward`: the latter governs every nonterminal state
while the former is a boundary condition.

Before the canonical reward gauge was introduced, the default library retained
a relative off-target desirability of `exp(-18)`. That positive value produces
finite leakage from every selected subgoal task into all other subgoal
boundaries. It remains available for historical reproduction by passing
`off_target_value=np.exp(-18.0)` explicitly, but is no longer the canonical
default. `from_desirabilities(...)` records the three construction values as
metadata;
`TaskLibrary.from_matrix(...)` instead accepts any validated finite,
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

[`Task.plan`](../src/andrew_mlmdp/hierarchy/model.py) delegates to
[`_compose_plan`](../src/andrew_mlmdp/hierarchy/model.py).
Planning performs four transformations.

First, reward inpainting converts the change requested by abstract control
into Layer-1 boundary rewards for every possible abstract outcome, including
the physical goal:

```math
r^1_t = \beta\,(u^2_t - p^2_t).
```

The active physical-goal reward is the same probability-difference term, not
`goal_reward`. Exponentiating these rewards with `lower_control_cost` produces
a target boundary desirability `z_target`. The configured `goal_reward` only
multiplies the single-boundary upper and goal-only desirabilities by a common
factor. Column normalization cancels that factor, so `goal_reward` does not
change upper termination, the goal-only policy, or trajectory likelihood.

Second, the target is expressed in the reusable task basis:

```math
w_{raw} = Q_b^+ z_{target},
\qquad
w = \max(0, w_{raw}).
```

`Q_b^+` is the pseudoinverse. For the canonical identity library,
`w_raw = z_target`, so every coefficient is positive, clipping is a no-op, and
the boundary target is reconstructed exactly. For a custom library with
positive off-target entries, an inpainted target can lie outside the
non-negative cone of `Q_b`; its exact linear coefficients then include
negative values and clipping introduces an approximation. With
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
preserves the coefficients produced by pseudoinversion and clipping; for the
identity library this reconstructs the target exactly. `c < 1` is more
diffuse, and `c > 1` is more competitive. Fractional powers are evaluated only
at strictly positive weights, preserving exact zeros and finite inactive-entry
gradients. The diagnostic hard winner-take-all mode preserves subgoal mass,
splits exact ties, and cannot be used with Adam.

Third, the lower desirability is composed linearly:

```math
z_i^1 = Z_i w,
\qquad
z_b^1 = Q_b w.
```

Finally, the standard LMDP reweighting formula converts this desirability into
`plan.lower_policy`. The complete implementation is in
[`_compose_policy`](../src/andrew_mlmdp/hierarchy/model.py).

The most useful debugging fields on `Plan` are
`upper_passive`, `upper_policy`, `rewards`, `raw_weights`,
`weights`, `desirability`, and `lower_policy`.

## 9. Execute the coupled process

[`_run_rollout`](../src/andrew_mlmdp/hierarchy/rollout.py) uses the
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
solved as a single-boundary first-exit task. Its configured `goal_reward` sets
only a common desirability scale and therefore cancels from its normalized
policy.

With `goal_learning="online"`, that column is replaced by a learned vector,
initialized to zero or supplied through `initial_goal_desirability`. Each
nonterminal physical move applies the requested number of fixed-point sweeps:

```math
z_i \leftarrow q_i(P_{II}^{T}z_i + P_{BI}^{T}z_b).
```

[`desirability_step`](../src/andrew_mlmdp/lmdp.py) performs one sweep. The
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
fixes the NMF scale gauge using the configured profile normalization and
absorbs the scale into row `j` of `W`. Peak normalization is the default;
`profile_normalization="l2"` selects the unit-L2 gauge. Reconstruction is
unchanged under either rescaling. Post-normalization objective descent is
therefore verified empirically rather than assumed from the standard MU
guarantee.

[`discover_subgoals`](../src/andrew_mlmdp/discovery.py) fits every
requested rank once, and `study.result(k)` retrieves that cached fit.
Regularized results expose the initial full objective and one value per
iteration through `objective_history`. The existing
`reconstruction_error` remains normalized generalized KL only, with no
smoothness penalty. The returned normalized `D` can then be passed to
`SubgoalBasis.from_profiles`, optionally applying the execution gate
described above. For an explicit L2 configuration, pass the same
`profile_normalization="l2"` option to discovery and basis construction.

## End-to-end implementation map

| Stage | Public result | Implementation |
| --- | --- | --- |
| Physical passive dynamics | `environment.passive` | `lmdp.passive_dynamics` |
| Flat goal solution | `Solution` | `Environment.solve` |
| Reusable subgoal representation | `SubgoalBasis` | `SubgoalBasis.from_locations` / `from_profiles` |
| Goal-conditioned construction | `Task` | `Template.task` -> `_build_task` |
| First-hit abstraction | `first_hit` | `_fundamental_matrix`, `_upper_dynamics` |
| Upper LMDP | `upper_desirability`, `upper_controlled` | `_solve_upper` |
| Reusable lower solutions | `task_basis` | `_task_basis` |
| Top-down composition | `Plan` | `compute_plan` -> `_compose_plan` |
| Coupled execution | `Rollout` | `_run_rollout` |
| Distributed discovery | `NMFStudy` | `discover_subgoals` |

No stage assumes a particular maze size or fixed number of subgoals. Array
shapes are derived from the supplied maze, goal partition, and profile count.

## Hierarchy diagnostics and visualization

Numerical diagnostics live in `andrew_mlmdp.hierarchy.diagnostics` and plotting
wrappers are exported from `andrew_mlmdp.plotting`. The numerical results copy
their arrays and make them read-only. Passing a template constructs an
uncached goal task, so diagnostic inspection does not populate the template's
task cache.

The access graph exposes three explicitly named arrays:

- `source_profiles`, copied from `basis.profiles`;
- `gated_profiles`, copied from `basis.access_profiles`; and
- `access_probabilities`, mapped directly from
  `task.subtask_access` through `task.interior_states`.

Peaks and centroids derived for graph layout are called `positions`.
They are visual positions only, never inferred physical entry states.

```python
from andrew_mlmdp import plotting
from andrew_mlmdp.hierarchy import (
    composition_trace,
    continuation_policies,
    upper_graph,
    sample_rollouts,
)

access = upper_graph(task, start_state=start)
continuations = continuation_policies(task)
weights = composition_trace(task, start_state=start)

plotting.plot_upper_graph(task)
plotting.plot_upper_policy(task, start_state=start)
plotting.plot_continuation_policies(task)
plotting.plot_composition_weights(task, start_state=start)

ensemble = sample_rollouts(task, start, seed=0)
plotting.plot_rollout_distribution(task, start, ensemble=ensemble)
plotting.plot_routes(task, start, ensemble=ensemble)
```

`ContinuationPolicy` stores the stationary `Plan` and the exact
refractory-adjusted columns produced by the rollout engine. A plot can show the
literal first post-access distribution only when the caller supplies an
explicit physical entry coordinate for that subgoal; display coordinates are
never substituted.

Composition diagnostics expose the actual implementation trace
`raw_weights -> clipped_weights -> weights`. The middle vector is
recorded by `Plan` at the point where it is passed into composition,
rather than reconstructed by diagnostics.

Trajectory length always means the number of physical steps. For model
rollouts this is `rollout.physical_steps`, equal to
`len(rollout.trajectory) - 1`; abstract accesses add no physical steps.
Repeated physical coordinates still count as sampled physical transitions.
Successful-route statistics exclude censored or failed outcomes while status
counts retain every rollout.

The NumPy object supplied to diagnostics must already contain the desired
fitted parameter values. `FitResult` snapshots are not applied to
templates implicitly.

## Differentiable hierarchical likelihood and fitting

The NumPy hierarchy remains the rollout and regression implementation. The
Torch likelihood rebuilds every parameter-dependent quantity in float64 for
each forward graph, while retaining the `P[next_state, current_state]`
orientation and exact latent controller-mode semantics. Only topology,
indices, physical passive dynamics, fixed normalized subgoal profiles, and
their normalization mode are safe to retain across optimizer steps. Gated
access profiles, lower and upper
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
exponent defaults come from the basis rather than unused gate fields on
`Parameters`. The hard gate is intentionally piecewise differentiable.
Entries below threshold have no local branch gradient, although gradients from
active entries may move the global threshold enough to activate them later. If
only exact profile peaks remain active, threshold and exponent can be weakly
identified or have zero gradients because the gated column remains the same
one-hot vector under either normalization mode.

For a set of physical goals `G`, the complete hierarchy is structurally
defined only below

```math
\tau_{max}=\min_{g\in G,j}\max_{s\ne g}D_{sj}.
```

`Template.threshold_range(goals)` reports this strict bound and
all limiting `(goal, subgoal_index)` pairs. Goal-task construction and fitting
reject public initial thresholds outside the corresponding domain. During
fitting, the private raw transform maps into
`(DOMAIN_EPS, nextafter(tau_max, -inf))`; the one-ULP upper margin prevents a
saturated sigmoid from eliminating the final support state. This changes
neither the public physical threshold nor the hard-gate equation.

`fit_parameters` minimizes the negative summed trajectory
log-likelihood using a private constrained parameterization. It never mutates
`Parameters`, `SubgoalBasis`, `Template`, or NumPy caches. Its
`best_values` snapshot can be passed explicitly to
`total_log_likelihood`. NumPy rollout from fitted
values requires separately constructing a fresh basis and template.

The fixed task library and `composition_exponent` are not Adam variables.
Adam fitting currently fixes `composition_exponent = c = 1.0` and rejects a
template configured with another value. The canonical reward gauge uses
`interior_reward = -1` and `goal_reward = 0`. Both remain configurable
construction inputs, but Adam holds their configured values fixed and rejects
selecting them. The identifiable reward-cost groups are
$\rho_1 = 1 / \lambda_1$, $\rho_2 = 1 / \lambda_2$, and
$\beta / \lambda_1$.

`fittable_parameters(template)` returns `lower_control_cost`,
`upper_control_cost`, `alpha`, and `beta`, plus `core_threshold` (`tau`) and
`core_exponent` (`eta`) for an active gated basis. The former behavioral
`off_target_reward` no longer exists. `off_target_value` is an optional
structural boundary leakage value, defaults to zero, and is never repurposed as
a behavioral or fitted reward. The historical `exp(-18)` value must be
requested explicitly.

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

Optimization plateau scales are independent of `tolerance`.
`scheduler_tolerance` is passed to `ReduceLROnPlateau`, while
`convergence_tolerance` defines meaningful best-loss improvement for
patience at the minimum learning rate. Omitting either option falls back to `tolerance`, allowing plateau decisions
to use a scientifically interpretable likelihood scale.


## Naming conventions

Names use the nearest module as context. Inside `andrew_mlmdp.hierarchy`, prefer
`Template`, `Task`, `Plan`, and `TaskLibrary`; repeating `Hierarchy` or
`LayerOne` does not add information there. Result records name the concept they
contain (`PairDiagnostics`, `UpperGraph`, `RolloutSummary`) rather than ending
in the generic suffix `Data`. Functions use a direct verb or noun phrase:
`diagnose_pair`, `composition_trace`, `sample_rollouts`, and `fit_parameters`.
Local sizes use `n_states`, `n_modes`, and similar `n_*` names.

These names are the sole project API; no alternate names are retained.
