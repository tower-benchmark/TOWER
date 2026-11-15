# Task layout staging

Each task currently has one USDA descriptor:

```text
tasks/tower_transfer_2.usda
tasks/tower_transfer_3.usda
tasks/cross_tower_transfer_2.usda
tasks/cross_tower_transfer_3.usda
tasks/aligned_stack_2.usda
tasks/aligned_stack_3.usda
```

These descriptors reference the six-block station foundation and record the
task name and parameter. They do not contain the upper blocks, goals or physics
needed for benchmark evaluation. Use the fixed scenes and verified assets in
the [public benchmark guide](../../docs/reference/bench-public.md). The CPU
backend validates descriptor names and asset closure; it does not provide real
tower physics.
