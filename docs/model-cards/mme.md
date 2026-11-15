# MME model card — draft

## Identity and intended use

MME, checkpoint at training step 60,000 with observation history. Intended for research evaluation of tower manipulation policies.
TOWER does not bundle these weights. Serving belongs
in XPolicyLab; a draft card does not establish available integration or weights.

TODO(owner): Confirm checkpoint identity, digest, distribution location and supported adapter.

## Training data

The checkpoint is associated with tower-manipulation demonstrations. Exact
training-data composition and rights require owner confirmation; this card makes
no claim of independent test data or broader robot generalization.

TODO(owner): Confirm high-level dataset composition, provenance rights and evaluation overlap.

## Evaluation

Target benchmark revision: **2026.11.1**. No new measurements were run for this card.

TODO(owner): Supply per-task and aggregate results, sample counts, uncertainty, runtime and checkpoint identities for 2026.11.1.

## License

TODO(owner): Confirm weight license, base-model obligations and training-data redistribution restrictions.

## Limitations

Performance depends on embodiment, camera layout, action timing and instruction
format. Simulation results do not establish real-hardware safety or transfer.
Do not use for unattended hardware control. Follow the [control safety contract](../guides/eef-control.md).
Paper measurements cannot substitute for evaluation of these release bytes.
