# Backgammon Observation Tensor Baseline

**Tensor Shape / Size:** 204 Dimensions (1D Vector Array)
**C++ Generation Logic Location:** `BackgammonState::ObservationTensor()`

## Section 1: Board Point Map (Indices 0 - 191)
*Exactly 24 operational Points actively bounding the board sequence map.*
*Using 4 precise constraint indices per point, per player. (192 slots structurally consumed here)*

For each operational point `(Idx: 0` to `23)`:
*   **Current Player Checker Density Logic:**
    *   `[Current + 0]`: Assigned `1.0` if exactly 1 checker sits on this point (0.0 otherwise).
    *   `[Current + 1]`: Assigned `1.0` if exactly 2 checkers sit on this point (0.0 otherwise).
    *   `[Current + 2]`: Assigned `1.0` if exactly 3 checkers sit on this point (0.0 otherwise).
    *   `[Current + 3]`: Assigned integer calculation `(Count - 3)` rendering >3 checker overflows (0.0 otherwise).
*   **Opponent Checker Density Logic:**
    *   `[Current + 4]`: Assigned `1.0` if exactly 1 checker sits on this point (0.0 otherwise).
    *   `[Current + 5]`: Assigned `1.0` if exactly 2 checkers sit on this point (0.0 otherwise).
    *   `[Current + 6]`: Assigned `1.0` if exactly 3 checkers sit on this point (0.0 otherwise).
    *   `[Current + 7]`: Assigned integer calculation `(Count - 3)` rendering >3 checker overflows (0.0 otherwise).

## Section 2: Global Environmental Parameters (Indices 192 - 203)
*The final 12 sequential slots dynamically monitoring precise global match logic conditions at generation scale.*

*   `[192]` - **Player Bar Block Limit**: Raw integer tracking pure metric count representing Player checkers physically relegated to the bar element.
*   `[193]` - **Player Scored / Bound Limit**: Raw integer rendering accurate tally count of all Player arrays successfully navigated Off-Board (Borne off).
*   `[194]` - **Player Engine Phasing Indicator**: Defined `1.0` strictly indicating it is physically the Player's turn (evals `0.0` conversely).

*   `[195]` - **Opponent Bar Block Limit**: Raw integer tracking opposing raw scale representing Opponent checkers relegated to the bar.
*   `[196]` - **Opponent Scored / Bound Limit**: Raw integer tally of Opponent checkers completing the map sequences Off-Board successfully.
*   `[197]` - **Opponent Engine Phasing Indicator**: Defined `1.0` mapping exclusively when it is currently the Opponent's sequence phase internally.

*   `[198]` - **Die 1 Parameter Scale / Output**: Pure internal pip sequence for the natively designated first physical operational die block element (defaults to 0 if dice are totally absent / complete).
*   `[199]` - **Die 2 Parameter Scale / Output**: Assigned raw mapping designating pip constraint integer values specifically tracking the sequential second dice matrix constraint natively.

*   `[200]` - **Moves Remaining Check (1)**: Categorical structural indicator validating explicitly `1.0` actively identifying `moves_remaining` exactly == 1.
*   `[201]` - **Moves Remaining Check (2)**: Categorical structural indicator validating explicitly `1.0` actively identifying `moves_remaining` exactly == 2.
*   `[202]` - **Moves Remaining Check (3)**: Categorical structural indicator validating explicitly `1.0` actively identifying `moves_remaining` exactly == 3.
*   `[203]` - **Moves Remaining Check (4)**: Categorical structural indicator validating explicitly `1.0` highlighting sequences mapping `moves_remaining` exactly == 4. *(Explicit Regression Security fix actively proving 4-move Doublet operations natively process array scale correctly without splitting sequences.)*
