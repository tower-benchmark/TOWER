# Scoring reference

`tower score` selects `bench-2026.11` for benchmark tasks. `paper-v1` remains a
public compatibility ruleset; it is not a distributed simulator harness.
The rule table below lists its differences. Use the benchmark ruleset for new
benchmark reports.

## Scoring

The versioned detector emits successful G/P operations, their sample indices, operation windows, disturbed block identities and source SHA-256 hashes. Paper Appendix B.1 defines **C = G + P**, **E = C + 1 − s**, and disturbance as non-target center translation **greater than 10 mm**, once per block per operation. Foundations, tools and both arms' current targets are excluded. Each arm's observed target interactions are recorded in `target_intervals`, independently of scoring windows and grasp-success confirmation. A former target's continuing motion remains excluded until rest or renewed contact from another manipulation, through moving fingers or the other operation's moving target. If the block was already moving when selected as a target, preceding collateral motion remains eligible. Physical release ends the held-target exclusion even when placement fails. Counts do not require a completed prefix; task success additionally requires the prescribed order and terminal arrangement. Safety stops remain valid failures; evaluator errors are excluded.

- **Task B (two-block transfer; task1) / Task A (cross transfer; task2):** acquisition requires holding, co-motion and lifting; placement requires release, the prescribed layer/orientation and physical support. Task 1 also checks terminal paired spacing.
- **Task E (layerwise pushdown; task4):** directed pushes require at least 5 mm of hand and block advance during a proximity interval, aligned hand/block motion, and full-block clearance beyond the source footprint along the block's long axis. Evidence uses 25 mm hand proximity confirmed for 0.1 s, 0.1 s clearance, and at most 0.5 s coasting after the last driven motion. There are **6/9 total target pushes** for two/three layers: the first 3/6 pushes are G and the last three are P. This matches the 2026.11 scorer and paper Task E; legacy generic transfer-target fields do not define this total. Passive departures beside stationary hands receive no credit.
- **Task D (lateral insertion; task5):** placement requires containment between the recorded slot walls, the authored insertion pose and preserved surrounding structure.
- **Task C (extract and stack; task6):** G requires holding while the entire block clears the source volume, laterally or above the roof, including initially protruding blocks. Roof placement permits shifted, intact supports and checks support, orientation and the paper's ±15 mm height tolerance.

- **Task F (tool-assisted extraction; task7):** each cycle scores tool acquisition and two tool-assisted pushes as three G steps, tool return as one P step, and two grasp–extraction/placement pairs as two G and two P steps. Totals are **5G/3P for two layers** and **10G/6P for three layers**. The authored prototype scene identifies `pusher_upper`/`pusher_lower`, the ordered side-block targets, opposite-arm transfers and destination goals. Pushes require closed-hand tool transport, contact near the tool's handle tip, directed hand/tool/block motion, and a graspable protrusion while the block remains in its source layer. Full extraction, released placement, ordered cycles, returned tools and retained source centers/roof determine completion. Tool bodies are excluded from D; each pushed block is the current target of its push step.

Task 7 uses a 25 mm exposed end as the numerical proxy for graspable protrusion, the prototype tool's local handle tip `(0, -0.166, 0.010)` m, and the shared 5 mm directed-motion and 0.1 s push-confirmation rules. Tool return requires release within 10 mm/10° of its initial pose and 0.3 s stability; a half-turn is not an equivalent pose for the asymmetric tool. These choices implement the paper's visually judged tool criteria. Block placements retain the paper's 10-second dwell.

All block placements require observed closed-hand carrying followed by release, **10 seconds after release**, continuous geometry and stability. Carrying evidence uses co-motion and limited relative slip independently of the G force proxy; passive arrival at a goal receives no P credit. Operation windows include withdrawal and retries until the next operation starts. Credit is not extrapolated beyond the recording. Previously completed operations retain credit if the terminal structure subsequently fails. Evidence after the scorer's generic 420-second limit is not scored. November 2026
rollouts use the shorter per-task deadlines in the [task registry](../concepts/tasks.md),
which are checked against the benchmark protocol.

Numerical choices for criteria judged visually in the paper are recorded in every result through `Rules`: 15 mm grasp surface proximity, 0.3 s holding, 70 mm maximum grasp opening, bilateral jaw effort ≥0.5 N for ≥80% of the hold window, ≤0.1 s effort gaps, ≤10 mm relative motion, ≥5 mm carrying motion/lift, 75 mm release opening or 40 mm clearance, 10° orientation tolerance, 20 mm paired-spacing tolerance, 8 mm support-height tolerance, ≥20% footprint support with center of mass over the contact hull, 10 mm insertion/structure position tolerance, and 0.5 mm slot-face tolerance. Stability requires ≤2 mm anchor translation, ≤2° anchor rotation, ≤5 mm/s linear velocity and ≤0.05 rad/s angular velocity. These are detector parameters; the paper's 10-second, 10-mm disturbance and 15-mm roof-height criteria are applied directly.

Disturbance attribution uses the recorded distal finger geometry and oriented block boxes with a 2 mm contact-proximity margin. A renewed interaction requires contact onset or a participant starting to move; changing the operation label alone is insufficient. Motion greater than 0.25 mm over 0.2 seconds identifies moving participants, and 0.3 seconds of quiet separates movement bouts. These attribution parameters do not change the **>10 mm** disturbance threshold.

## Input paths

The default `files` path reads persisted evidence. The compatibility input-path
value `campaign` reconstructs status and omits distal corner evidence. These are
different input contracts; selecting a ruleset does not change the input path.

| Contract | `campaign` | `files` |
|---|---|---|
| Constructor | `from_campaign_rows`; disk adapter `from_campaign_files` | `from_files` (`load_recording` alias) |
| Input hashes | Empty mapping | SHA-256 of six recording files |
| Distal finger corners | Absent | Transformed measured link corners |
| Status | Reconstructed `duration_reached`, valid, returncode zero | Persisted `status.json` |
| Compatibility status amendment | Copy `task_failure` only, except Task1-2 | No reconstruction |
| Time coverage | Contiguous rows starting exactly at zero, at most 0.1 s gaps | Existing file validation, final-state coverage, configured time-limit filtering |

Reports record `ruleset`, `input_path`, `tower_eval_version` and
`canonical_task_id`. Do not compare different input contracts as equivalent.

## Rulesets

These overrides take precedence over the generic detector criteria above.
Two-block transfer, 2 layers, uses the four-place hold rule
(`tower-task12-four-place-hold-v1`); 3 layers routes `tower-sim-trajectory-v5`.

| Rule | `paper-v1` | `bench-2026.11` |
|---|---|---|
| Task5 receiving tower | `destination` | `source` |
| Task5 side walls | Names containing `side` | Non-target, non-fixed members of receiving layer |
| Task5 thresholds | Original defaults | 0.5 s dwell, 5 mm / 5° pose, 0.5 mm / 0.5° stability, 0.01 m/s / 0.1 rad/s velocity |
| Task5 measured gate | Absent | Measured containment, structural stability and latched failure gate; `task5-measured-insertion-v1` |
| Task6 legacy untouched gate | Structural XY within foundation; original height/orientation gates | Accumulated translation ≤2 mm and rotation ≤5°; retains height/orientation gates |
| Task6 terminal success | Legacy grasp/place/order/terminal decision | Final released roof support and no fall anywhere; legacy G/P/D retained |
| Task4 terminal success | Legacy directed push/order/clearance decision | Final source-base removal using SAT or table contact polygon; legacy G/P/D retained |
| Task1-2 success | Original dwell, pairing and order checks | `FOUR_PLACE_HOLD`: first eligible whole-second success prefix; pairing/order not required for success |
| Task7 push direction | Fixed layer directions | Metadata direction when `evaluation_task` starts with `task7-`; otherwise historical fallback |
| Task7-3 structural survival | Current support-height/orientation/footprint condition | Entire-recording collapse veto: loss below half minimum block dimension and tilt below 45°, latched |
| Optional disturbance computation | Historical API always computes it | In-memory `include_disturbances=False` used by stop detectors |
