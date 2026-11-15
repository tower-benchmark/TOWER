# Third-party notices

First-party TOWER code is Apache-2.0. The following components retain their
own licenses and attribution.

- **XPolicyLab** (Apache-2.0), pinned in `third_party/XPolicyLab`.
  Upstream: <https://github.com/XPolicyLab/XPolicyLab>. Its LICENSE and
  policy-specific notices remain in the submodule.
- **AgileX agx_arm_urdf**, revision `680fa925cd`: MIT,
  Copyright (c) 2026 aalicecc. Public source:
  <https://github.com/agilexrobotics/agx_arm_urdf/tree/680fa925cd>.
  The complete license is beside the frozen URDF/xacro files at
  `packages/nero-kinematics/src/nero_kinematics/models/agilex/LICENSE`.
  Asset notices are also retained in `hf-assets/THIRD_PARTY_NOTICES.md`.
- **eef_ik_movejs / Nero EEF IK**: a locally modified TOWER fork of source
  distributed under Apache-2.0,
  with AgileX model data (MIT). See the adjacent
  [model notices](packages/nero-kinematics/src/nero_kinematics/models/THIRD_PARTY_NOTICES.md)
  and LICENSE files. Model geometry and calibration values are unchanged.
- **piper_ros algorithm reference**, revision
  `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`: MIT,
  Copyright (c) 2024 RosenYin. TOWER's solver is an independent implementation;
  no Piper implementation was copied. Source:
  <https://github.com/agilexrobotics/piper_ros/blob/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/src/piper/scripts/piper_pinocchio/piper_pinocchio.py>.
  Full terms: [LICENSE.piper_ros](packages/nero-kinematics/LICENSE.piper_ros).

- **RoboDojo** (<https://github.com/RoboDojo-Benchmark/RoboDojo>), commit
  `ee67a1468510da7624a089164402359f2afc72c8`: MIT. The default joint-target
  interpolation in `tower_eval/controllers/controller.py` reimplements the
  behavior of RoboDojo's `process_control_info` (linear ramp over the first 80%
  of control steps, then hold). No RoboDojo source is vendored.

### Vendor model comment redactions

The vendor URDF/xacro files retain AgileX revision `680fa925cd` non-comment bytes
identically. Their XML comments differ only to redact personal identifiers;
geometry, inertial data, joints and limits are unchanged.
