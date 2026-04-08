# Development Log: feature/doublets-and-dmp

## 1. 4-Move Doublet Horizon Expansion
- **Action Space Escalation**: Increased `kNumDistinctActions` to `913,952`, permanently scaling maximum action packing constraints to handle Base-26 base multipliers to handle continuous 4-move atomic doublet sequences instead of splitting them.
- **`DoApplyAction` / `UndoAction` Refactoring**: Updated game loops to extract lengths dynamically from sequential loops rather than enforcing fixed 2-move ceilings.
- **Notation Overhaul**: Modernized `ActionToString` loop tracking to accurately group repeating 4-move subactions using multipliers (e.g., `24/20(4)`), explicit hit notation `*`, and normalized string labeling for Off/Bar board configurations.
- **Dice Buffer Synchronization**: Re-engineered `SetDice` functionality to automatically inject four explicitly distinct copies of the current roll into `dice_` on any doublet to fully supply the 4-atomic pip requirements.

## 2. Double Match Point (DMP) Refactoring
- **Parameter Architectures**: Appended `dmp_only` boolean runtime parameter into `BackgammonGame` dynamically configurable through the `GameParameter` dictionary map schemas via PyBind Python abstractions (default: `false`).
- **Reward Flattening Enforcement**: Altered `BackgammonState::Returns()` to strictly override state gammon/backgammon multipliers with zero-sum flattened `{1.0, -1.0}` utility outputs exclusively representing Win/Loss sequences when `dmp_only` is toggled.
- **Maximum Utility Update**: Patched scaling bound checker `MaxUtility()` logic to recognize constraints dropping entirely to a strict boundary of `1.0` ensuring Python arrays properly normalize returns logic.
- **Checker Deprecation Security**: Repaired dynamic checks inside `Returns` and `IsTerminal` loop tests by intentionally deprecating historically hardcoded integers (`15` checkers) transitioning to secure parameter pulls using `NumCheckersPerPlayer()`.

## 3. Observation Tensor Alignments
- **Moves Tracking Regression (Bug Fix)**: Eliminated legacy code branching enforcing sizes limited strictly to 2 nodes. Re-engineered `moves_remaining` computations purely aligning to raw pulls computing `dice_.size()` resolving a bug dropping doublets to size 2 internally.

## 4. Quiet Test Suite Bugs Addressed (Test Suite Integrity Fixes)
- **Coordinate System Overrides (Phantom Bear-offs)**: Early stress verification tests failed out-of-the-gate securely assigning parameter testing to place `X` checkers precisely at point `0`. Because standard games orient board mapping upwards toward points `18-23`, running a 2 pip move pushed elements mathematically to point `2` circumventing termination states. Re-centered logic structurally placing `X` arrays natively in the home board at point `23` resolving false-negative triggers and bearing tests properly terminating mathematically past `< 24`.
- **Gammon Testing Output Defaulting Conflicts**: `RegressionGammonTest` consistently failed asserting target `{2.0, -2.0}` because the backend `ScoringType` strictly defaults out to `kWinLossScoring` without explicit arrays attached. Forced the test parameter dictionaries to map `full_scoring` internally successfully rendering test verifications compliant with gammon logic outputting raw multiplier rewards securely yielding the appropriate payout structures.

## 5. Action Space Optimization & Mapping
- **The Mathematical Derivation**: The base size of elements is computed physically as `26^4` (456,976), representing Base-26 slots across 4 moves for continuous atomic nodes. The overall space explicitly scales to `2 * 26^4` (913,952) physically doubling the mapped output limits.
- **The Die-Order Offset**: The 913,952 mapping mathematically separates "High-Die First" sequences vs. "Low-Die First" sequencing vectors to unambiguously resolve intermediate hitting and blocking states across asynchronous subaction generation. "High-Die" moves map strictly 0 to 456,975; "Low-Die" moves push an absolute mapping logic bounded at +456,976 targeting the sequential upper half. 
- **Doublet Exception**: While the Action Space mathematically reserves bounds scaling for upper-limit low-first operations, exact doublets (eg. 4-4-4-4) systematically nullify index requirements offsetting constraints due to mathematically equal pip allocations across sequences.
- **Reachability (DMP Impact)**: The Double Match Point (`dmp_only`) mode natively functions as a 1D utility flattening algorithm having exactly zero architectural impact on physical Backgammon mechanics. Consequentially, both structural halves of the OpenSpiel Action Space natively remain actively populated and functionally required for sequential physical moves inside all permutations of DMP architecture.
