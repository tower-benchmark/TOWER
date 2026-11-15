---
license: other
license_name: other
---

# TOWER-Assets — legacy 0.0.2 staging manifest

Asset inventory retained for the earlier TOWER asset snapshot. This
source directory contains no binary assets. The current public dataset is
[tower-benchmark/TOWER-Assets](https://huggingface.co/datasets/tower-benchmark/TOWER-Assets).
Use `python tools/simulator/benchmark_assets.py download --assets simulator-assets`
from the repository root for the pinned November 2026 release. This legacy
inventory describes a different asset layout; see the [current guide](../docs/reference/bench-public.md).

This legacy manifest remains because `tower_eval` example configs reference
`hf-assets/manifest.json` and CI checks its manifest closure. It is not the
November 2026 benchmark asset inventory. Use the downloader above for the
current release and its pinned asset lock.

## Contents

```text
station/station.usdz
blocks/block_01.usd ... blocks/block_24.usd
blocks/texture_variant_00.usda ... texture_variant_07.usda
blocks/base/block_base.usd
blocks/base/textures/pbr/*.png
blocks/textures/variant_00/*.png ... variant_07/*.png
tasks/tower_transfer_2.usda
tasks/tower_transfer_3.usda
tasks/cross_tower_transfer_2.usda
tasks/cross_tower_transfer_3.usda
tasks/aligned_stack_2.usda
tasks/aligned_stack_3.usda
```

This inventory describes the table, robots, cameras and six foundation blocks.
Its task USDA files are descriptors, not complete physics layouts. Use the
current benchmark's asset and scene manifests for evaluation.

## Licensing

TOWER-authored block assets are licensed under CC BY 4.0. The station includes
an AgileX arm model licensed under MIT; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The per-file
[`manifest.json`](manifest.json) is authoritative.
