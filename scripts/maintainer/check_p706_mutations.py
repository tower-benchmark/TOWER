"""Prove all reviewed construction drifts fail the tower-sim suite.

Run in an isolated clean worktree. Sources are restored even after test failure;
bytecode is isolated per run so same-size edits cannot reuse stale cached code.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "packages/tower-sim/src/tower_sim/bench"
# Removed alias/layout mutations are documented in the purge evidence inventory.
MUTATIONS = (
    (
        "startup-sdk-completion-argv",
        "app.py",
        "for flag in RENDER_COMPLETION_ARGS",
        "for flag in ()",
    ),
    (
        "startup-parent-evaluation-finalize",
        "../../../../../tools/simulator/benchmark_candidate.py",
        'finalize(cache_options, Path(request["output"]), passed=code == 0)',
        "pass",
    ),
    (
        "startup-parent-smoke-finalize",
        "../../../../../scripts/maintainer/public_smoke.py",
        "finalize(args, args.output, passed=passed)",
        "pass",
    ),
    ("startup-warmup-success", "shader_cache.py", "        if passed:", "        if True:"),
    (
        "startup-parent-cleanup",
        "shader_cache.py",
        "        shutil.rmtree(directory, ignore_errors=True)",
        "        pass",
    ),
    ("startup-completion", "bootstrap.py", "zero_delay=True", "zero_delay=False"),
    (
        "startup-replay-cache",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "shader_cache=self.shader_cache",
        "shader_cache=None",
    ),
    (
        "startup-cache-copy-config",
        "shader_cache.py",
        "seed=Path(seed).resolve(),",
        "seed=Path(output).resolve(),",
    ),
    (
        "startup-worker-cache",
        "../../../../../tools/simulator/benchmark_candidate.py",
        'runner.shader_cache = configuration(args, Path(request["output"]))',
        "runner.shader_cache = None",
    ),
    (
        "startup-smoke-cache",
        "../../../../../scripts/maintainer/public_smoke.py",
        "runner.shader_cache = configuration(args, work)",
        "runner.shader_cache = None",
    ),
    (
        "startup-child-cache",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "command.extend(child_arguments(cache_options))",
        "command.extend([])",
    ),
    (
        "startup-smoke-child-cache",
        "../../../../../scripts/maintainer/public_smoke.py",
        "command.extend(child_arguments(args))",
        "command.extend([])",
    ),
    (
        "public-smoke-entry-registration",
        "verification.py",
        '    "tools/simulator/benchmark_portable.py",\n',
        "",
    ),
    (
        "public-smoke-entry-bytes",
        "source_receipt.py",
        'and row.get("sha256") == expected["sha256"] == sha(path)',
        "and True",
    ),
    (
        "public-smoke-entry-marker",
        "source_receipt.py",
        '            and row.get("startup_sha256") == expected["sha256"]\n',
        "",
    ),
    (
        "public-smoke-entry-any-main",
        "source_receipt.py",
        "or (frozen_stem and not (known_package or known_entry))",
        'or (frozen_stem and name != "__main__" and not (known_package or known_entry))',
    ),
    (
        "public-smoke-startup-marker",
        "../../../../../tools/simulator/benchmark_portable.py",
        "__tower_entry_sha256__ = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()",
        "__tower_entry_sha256__ = None",
    ),
    (
        "asset-manifest-preflight-canonical-identity",
        "../../../../../scripts/maintainer/gpu_baseline.py",
        "asset_manifest_identity(sha(manifest)) != expected",
        "sha(manifest) != expected",
    ),
    (
        "public-expat-exact-native-root",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "expat.parent == native_stdlib",
        "True",
    ),
    (
        "public-expat-native-filename",
        "../../../../../scripts/maintainer/candidate_replay.py",
        'and expat.name in {"pyexpat" + suffix for suffix in EXTENSION_SUFFIXES}',
        "and True",
    ),
    (
        "public-expat-required-binding",
        "../../../../../scripts/maintainer/candidate_replay.py",
        'roots["pyexpat"] = roots["xml.parsers.expat"] = [str(expat)]',
        "pass",
    ),
    (
        "public-conda-record-binding",
        "verification.py",
        "                    verify_conda_record_entry(conda, path, payload, record, contents)",
        "                    pass",
    ),
    (
        "public-conda-ik-content-verification",
        "verification.py",
        "if expected is None or actual != expected:",
        "if False:",
    ),
    (
        "public-conda-ik-module-binding",
        "verification.py",
        "if not files or str(Path(module.__file__).resolve()) not in files:",
        "if False:",
    ),
    (
        "public-conda-ik-version-pin",
        "verification.py",
        "if module.__version__ != expected_version:",
        "if False:",
    ),
    (
        "purge-resource-adapter",
        "../backend/resources.py",
        "    return DATA / PATHS[name]\n\n\ndef default_path",
        "    return Path(checkout) / PATHS[name]\n\n\ndef default_path",
    ),
    (
        "reference-tag-refusal",
        "../../../../../tools/simulator/benchmark.py",
        "    if args.reference:\n        refuse_reference()\n",
        "",
    ),
    (
        "public-envelope",
        "../../../../../tools/simulator/benchmark_candidate.py",
        'if request.get("public") != envelope(release, benchmark=benchmark_version):',
        "if False:",
    ),
    (
        "public-pilot-review-binding",
        "../../../../../tools/simulator/benchmark_candidate.py",
        'read_bound(review["path"], review["sha256"])',
        'read_bound(review["path"])',
    ),
    (
        "public-posture-evidence-binding",
        "../../../../../tools/simulator/benchmark_candidate.py",
        'review["path"], *posture',
        'review["path"]',
    ),
    (
        "public-recorded-posture-gate",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "        posture = benchmark.posture_evidence(\n"
        '            read_bound(review["path"], review["sha256"]),\n'
        '            request["task"],\n'
        '            benchmark.read(request["package"])["package_digest"],\n'
        "        )\n",
        "        posture = []\n",
    ),
    (
        "public-ik-bounds",
        "../../../../../tools/simulator/benchmark_candidate.py",
        'if sha(request["ik_bounds"]) != sha(spec):',
        "if False:",
    ),
    (
        "public-preflight-release",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "    release = verify_release(\n"
        '        request["release_manifest"], checkout=ROOT, '
        'expected=request["release_manifest_sha256"]\n'
        "    )",
        '    release = dict(manifest_sha256=request["release_manifest_sha256"], '
        'commit=request["public"]["commit"])',
    ),
    (
        "public-source-closure",
        "evaluation.py",
        "return self.verification.verify_sources(sources)",
        "return sources",
    ),
    (
        "public-release-checkout-head",
        "verification.py",
        'if git(root, "rev-parse", "HEAD").decode().strip() != body["commit"]:',
        "if False:",
    ),
    (
        "public-release-clean-tree",
        "verification.py",
        'if git(root, "status", "--porcelain", "--untracked-files=no").strip():',
        "if False:",
    ),
    (
        "public-checkout-file-binding",
        "verification.py",
        'if sha(path) != expected:\n            raise ValueError(f"Changed release checkout',
        'if False:\n            raise ValueError(f"Changed release checkout',
    ),
    (
        "public-wheel-extra-payload",
        "verification.py",
        'raise ValueError(f"Unexpected release wheel member: {member}")\n        if not result:',
        "pass\n        if not result:",
    ),
    (
        "public-installed-file-set",
        "verification.py",
        "if loaded_files != set(members):",
        "if False:",
    ),
    (
        "public-release-data-containment",
        "verification.py",
        "        path = contained(root, relative)\n        contents = path.read_bytes()",
        "        path = (root / relative).resolve()\n        contents = path.read_bytes()",
    ),
    (
        "public-release-wheel-containment",
        "verification.py",
        'wheel = contained(path.parent, entry["wheel"])',
        'wheel = (path.parent / entry["wheel"]).resolve()',
    ),
    (
        "public-installed-containment",
        "verification.py",
        "loaded = contained(location.parent / module, relative)",
        "loaded = (location.parent / module / relative).resolve()",
    ),
    (
        "public-record-content-verification",
        "verification.py",
        "if actual != expected or (size and len(payload) != int(size)):",
        "if False:",
    ),
    (
        "public-ik-dependency-pins",
        "verification.py",
        "if name in IK_PINS and names != DEPENDENCIES and distribution.version != IK_PINS[name]:",
        "if False:",
    ),
    (
        "public-ik-native-stack",
        "verification.py",
        'names = tuple(IK_PINS) + ("scipy",) if args.ik else DEPENDENCIES',
        'names = ("numpy", "scipy", "casadi") if args.ik else DEPENDENCIES',
    ),
    (
        "public-ik-release-verification",
        "verification.py",
        "        verify_release(args.release, checkout=args.checkout, expected=args.release_sha256)",
        "        pass",
    ),
    ("public-ik-dependency-identity", "verification.py", "if ik != expected_ik:", "if False:"),
    (
        "public-release-request-binding",
        "verification.py",
        '            expected=request["release_manifest_sha256"],',
        "            expected=None,",
    ),
    (
        "public-dependency-request-binding",
        "verification.py",
        'read(request["dependency_manifest"], request["dependency_manifest_sha256"])',
        'read(request["dependency_manifest"])',
    ),
    (
        "public-ik-request-binding",
        "verification.py",
        '            request["ik_dependency_manifest"], request["ik_dependency_manifest_sha256"]',
        '            request["ik_dependency_manifest"]',
    ),
    (
        "public-assets-verification",
        "verification.py",
        'assets = self.verify_assets(request["assets"])',
        "assets = {}",
    ),
    (
        "public-replay-data-verification",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            verify_data(\n"
        "                self.root,\n"
        "                benchmark=self.benchmark_version,\n"
        "                assets=assets,\n"
        "                data_root=self.data_root,\n"
        "                v1_root=self.v1_root,\n"
        "            )\n"
        "            release = verify_release",
        "            release = verify_release",
    ),
    (
        "public-replay-release-verification",
        "../../../../../scripts/maintainer/candidate_replay.py",
        '            release = verify_release(self.root / "dist/release-manifest.json", '
        "checkout=self.root)",
        "            release = {}",
    ),
    (
        "public-setup-release-verification",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            self.release = verify_release(\n"
        '                self.root / "dist/release-manifest.json", checkout=self.root\n'
        "            )",
        "            self.release = {}",
    ),
    (
        "public-setup-no-reference",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "                reference=False,",
        "                reference=True,",
    ),
    (
        "public-entrypoint-release-binding",
        "../../../../../scripts/maintainer/candidate_replay.py",
        'expected = release["checkout_files"][relative]',
        'expected = hashlib.sha256(subprocess.check_output(["git", "-C", str(root), "show", '
        'f"HEAD:{relative}"])).hexdigest()',
    ),
    (
        "public-entrypoint-benchmark",
        "verification.py",
        '    "tools/simulator/benchmark_candidate.py",\n',
        "",
    ),
    (
        "public-diagnostic-score-default",
        "../../../../../tools/simulator/benchmark_diagnostic_score.py",
        'def score_physical_diagnostic(run, task, *, reference=False, benchmark_version="2026.11.1", '
        "v1_root=None):",
        'def score_physical_diagnostic(run, task, *, reference=True, benchmark_version="2026.11.1", '
        "v1_root=None):",
    ),
    (
        "public-seed-links-on-failure",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "    if not diagnostic:\n        # Preserve",
        "    if not diagnostic and not code:\n        # Preserve",
    ),
    (
        "public-smoke-child-environment",
        "../../../../../scripts/maintainer/public_smoke.py",
        "result = subprocess.run(command, env=public.child_environment(), check=False)",
        "result = subprocess.run(command, check=False)",
    ),
    (
        "public-formal-status-identity",
        "evaluation.py",
        "            status.update(self.verification.provenance)",
        "            pass",
    ),
    (
        "public-invalid-status-identity",
        "evaluation.py",
        "**(self.verification.provenance if self.verification is not None else {}),",
        "**{},",
    ),
    (
        "public-replay-status-identity",
        "replay.py",
        '                        **getattr(context, "public_provenance", {}),\n',
        "",
    ),
    (
        "public-summary-identity",
        "replay.py",
        '\n            **getattr(context, "public_provenance", {}),\n',
        "\n",
    ),
    (
        "public-final-import-closure",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            verify_sources(\n"
        '                {row["path"]: row["sha256"] for row in receipt["modules"].values()},\n'
        "                release,\n"
        "                allowed_roots=roots,\n"
        "            )",
        "            pass",
    ),
    (
        "public-formal-verifier-type",
        "policy_entry.py",
        "if verification is not None and type(verification) is not ReleaseVerification:",
        "if False:",
    ),
    (
        "public-ik-python-source-binding",
        "verification.py",
        'identity["python_sources"] = ik_python_sources()',
        'identity["python_sources"] = {}',
    ),
    (
        "public-untracked-ignored-runtime",
        "verification.py",
        'for ignored in ((), ("--ignored",)):',
        "for ignored in ((),):",
    ),
    (
        "purge-maintainer-option-abbreviation",
        "../../../../../scripts/maintainer/gpu_baseline.py",
        "    for command in sub.choices.values():\n        command.allow_abbrev = False\n",
        "",
    ),
    (
        "public-compatibility-source-binding",
        "../../../../../scripts/maintainer/candidate_release.py",
        "            for member in members:",
        '            for member in (m for m in members if not m.startswith("nero_eef_ik/")):',
    ),
    (
        "public-formal-source-verification",
        "verification.py",
        "        return verify_sources(\n            sources,",
        "        return (lambda *args, **kwargs: sources)(\n            sources,",
    ),
    (
        "public-shipped-alias-wheel",
        "verification.py",
        '("nero_kinematics", "nero_eef_ik")',
        '("nero_kinematics",)',
    ),
    (
        "public-alias-source-binding",
        "../../../../../scripts/maintainer/candidate_release.py",
        "            for member in members:",
        '            for member in (m for m in members if not m.startswith("nero_eef_ik/")):',
    ),
    (
        "public-stdlib-customization-binding",
        "verification.py",
        "        roots.add(str(customization.resolve()))",
        "        pass",
    ),
    (
        "replay-renderer-order",
        "app.py",
        "        app = factory(*args, **kwargs)",
        "        renderer_factory()\n        app = factory(*args, **kwargs)",
    ),
    (
        "replay-patch-visibility",
        "environment.py",
        "def renderer_base():\n"
        "    from tower_sim.replay.dual_replay_worker import _configure_calibration_renderer\n",
        "from tower_sim.replay.dual_replay_worker import _configure_calibration_renderer\n"
        "\n"
        "def renderer_base():\n",
    ),
    (
        "replay-closure-planted-generation",
        "../../../../../scripts/maintainer/check_eval_import_closure.py",
        "        if forbidden(fullname):",
        "        if False:",
    ),
    (
        "config-default-value",
        "../config.py",
        "gripper_finger_effort_limit_n: float = 22.0",
        "gripper_finger_effort_limit_n: float = 21.0",
    ),
    (
        "config-default-type",
        "../config.py",
        "reset_settle_seconds: float = 1.0",
        "reset_settle_seconds: int = 1.0",
    ),
    ("config-v1-refusal", "../_config_v1.py", "if self.max_recoveries != 0:", "if False:"),
    (
        "config-generation-edge",
        "environment.py",
        "from tower_sim.config import SimConfig as Config",
        "from tower_sim.generation.config import GenerationConfig as Config",
    ),
    (
        "replay-module-constant",
        "../replay/dual_replay_worker.py",
        "CALIBRATION_RENDER_WARMUP_FRAMES = 24",
        "CALIBRATION_RENDER_WARMUP_FRAMES = 24 + 1",
    ),
    (
        "replay-closure-loaded-name",
        "../../../../../scripts/maintainer/check_eval_import_closure.py",
        "            if forbidden(name):",
        "            if False:",
    ),
    (
        "replay-closure-checkout-path",
        "../../../../../scripts/maintainer/check_eval_import_closure.py",
        "            if path.is_relative_to(checkout):",
        "            if False:",
    ),
    (
        "replay-closure-frozen-path",
        "../../../../../scripts/maintainer/check_eval_import_closure.py",
        '            if "/simulator/benchmark-202611/" in path.as_posix():',
        "            if False:",
    ),
    (
        "replay-closure-isolation",
        "../../../../../scripts/maintainer/check_eval_import_closure.py",
        "    if not sys.flags.isolated:",
        "    if False:",
    ),
    (
        "replay-closure-script-location",
        "../../../../../scripts/maintainer/check_eval_import_closure.py",
        "    if Path(__file__).resolve().is_relative_to(checkout):",
        "    if False:",
    ),
    (
        "replay-closure-record-hash",
        "../../../../../scripts/maintainer/check_eval_import_closure.py",
        "member.hash.value != digest",
        "False",
    ),
    ("config-unknown-v1", "../config.py", "        if unknown:", "        if False:"),
    (
        "config-receipt-model",
        "source_receipt.py",
        '"tower_eval.simulator.isaac", "tower_sim.backend.model", "tower_sim.generation"',
        '"tower_eval.simulator.isaac", "tower_sim.generation"',
    ),
    (
        "config-receipt-generation",
        "source_receipt.py",
        'or name.startswith("tower_sim.generation.")',
        "or False",
    ),
    ("tcp-kinematics", "bootstrap.py", "key=key, kinematics=cache", "key=key, kinematics=None"),
    (
        "gravity-wrapper",
        "bootstrap.py",
        "prepare_scene=scene.prepare_physics_scene",
        "prepare_scene=lambda prepare, *a, **kw: prepare(*a, **kw)",
    ),
    ("robot-config", "bootstrap.py", 'robot_config=resources.read("robot")', "robot_config={}"),
    (
        "solver-iterations",
        "environment.py",
        "CreateSolverPositionIterationCountAttr(16)",
        "CreateSolverPositionIterationCountAttr(8)",
    ),
    ("renderer", "bootstrap.py", "renderer=app.renderer", "renderer=None"),
    ("cuda-visible-devices", "bootstrap.py", 'unset=("CUDA_VISIBLE_DEVICES",)', "unset=()"),
    (
        "renderer-controls",
        "app.py",
        'for key, value in resources.read("renderer_controls").items():',
        "for key, value in {}.items():",
    ),
    (
        "home-position-targets",
        "bootstrap.py",
        "        robot.set_joint_position_targets(target, indices=indices, joint_indices=names)\n",
        "",
    ),
    (
        "home-nonzero-velocities",
        "bootstrap.py",
        "robot.set_joint_velocities(np.zeros_like(target), indices=indices, joint_indices=names)",
        "robot.set_joint_velocities(np.ones_like(target), indices=indices, joint_indices=names)",
    ),
    (
        "initial-readback-tolerance",
        "bootstrap.py",
        "np.allclose(state.qpos, physical, atol=2e-6, rtol=0)\n"
        "            and np.allclose(measured_encoder, encoder, atol=2e-6, rtol=0)",
        "np.allclose(state.qpos, physical, atol=2e-2, rtol=0)\n"
        "            and np.allclose(measured_encoder, encoder, atol=2e-2, rtol=0)",
    ),
    (
        "gravity-all-to-any",
        "bootstrap.py",
        "np.asarray(robot.get_body_disable_gravity()).all()",
        "np.asarray(robot.get_body_disable_gravity()).any()",
    ),
    (
        "mass-readback-tolerance",
        "bootstrap.py",
        "measured, expected_mass, rtol=1e-5",
        "measured, expected_mass, rtol=1e-1",
    ),
    (
        "coupled-gripper-default",
        "bootstrap.py",
        'settings.get("coupled_gripper_fingers", False)',
        'settings.get("coupled_gripper_fingers", True)',
    ),
    (
        "task23-block-count",
        "scene.py",
        "len(current) != benchmark_task(task).block_count",
        "len(current) != benchmark_task(task).block_count + (task == 'task2-3')",
    ),
    (
        "transformation-arguments",
        "bootstrap.py",
        "update_transformations(False, True, True, False)",
        "update_transformations(True, True, True, False)",
    ),
    (
        "author-before-clone",
        "scene.py",
        "                result = super().clone(source_prim_path, prim_paths, **kwargs)\n"
        '                hook.report["clone_calls"] += 1\n'
        '                hook.report["usd_readback"] = hook.author(stage, hook.plans, hook.bond_path)',
        '                hook.report["usd_readback"] = hook.author(stage, hook.plans, hook.bond_path)\n'
        "                result = super().clone(source_prim_path, prim_paths, **kwargs)\n"
        '                hook.report["clone_calls"] += 1',
    ),
    (
        "viewport-updates",
        "app.py",
        '            "disable_viewport_updates": bool(config.get("headless", True)),\n',
        "",
    ),
    ("rendering-dt", "bootstrap.py", "rendering_dt=1 / 30", "rendering_dt=1 / 60"),
    (
        "gripper-command-limit",
        "bootstrap.py",
        "limits[arm * 8 + 7] = [0.0, 0.1]",
        "limits[arm * 8 + 7] = [0.0, 0.2]",
    ),
    (
        "fit-receipt-hash",
        "bootstrap.py",
        'if sha(resources.path("tcp_fit_receipt")) != fit_receipt_sha256:',
        "if False:",
    ),
    ("fit-receipt-status", "bootstrap.py", 'if fit_receipt["status"] != "passed":', "if False:"),
    ("config-validation", "bootstrap.py", "    cfg.validate()\n", ""),
    (
        "gpu-assembly-complete-coverage",
        "../../../../../scripts/maintainer/gpu_baseline.py",
        "    if present != set(tasks):",
        "    if False:",
    ),
    ("replay-physical-prevalidation", "replay.py", "    validate_trajectory(context, rows)\n", ""),
    (
        "replay-drive-source-route",
        "replay.py",
        "return context.collect_sources(source_collector)",
        "return source_collector()",
    ),
    (
        "replay-final-drive-receipt",
        "replay.py",
        "        context.finalize_drive_receipt()\n",
        "        pass\n",
    ),
    (
        "atomic-drive-receipt",
        "resources.py",
        "os.link(temporary, path)",
        "Path(path).write_bytes(Path(temporary).read_bytes())",
    ),
    (
        "reference-contact-windows",
        "contacts.py",
        "reference_windows=reference_windows",
        "reference_windows=False",
    ),
    ("serial-c19-range", "replay.py", "    if count >= 2:\n", "    if count >= 1:\n"),
    ("replay-settling-stage", "replay.py", "        settle(rows, env, contacts)\n", ""),
    ("replay-guard-enable", "replay.py", "        camera_guard.enabled = True\n", ""),
    (
        "replay-guard-consume",
        "replay.py",
        "rejected = camera_guard.consume(at_tick)",
        "rejected = {}",
    ),
    (
        "replay-shared-cache",
        "replay.py",
        'factories.scorer(task, mode="fast", computation=shared_cache)',
        'factories.scorer(task, mode="fast")',
    ),
    (
        "yaml-first-import-receipt",
        "source_receipt.py",
        "        sys.addaudithook(self.observe)\n",
        "",
    ),
    (
        "replay-request-id",
        "replay.py",
        'metrics.request_id(row["candidate"]["seed"], tick)',
        '"uninjected"',
    ),
    (
        "replay-qualification-call",
        "replay.py",
        "                qualify_capture(\n"
        "                    env.n, camera.report(), assignment, source_parity, sources, context.output\n"
        "                )",
        "                pass",
    ),
    (
        "replay-first-environment-only",
        "replay.py",
        "    validate_trajectory(context, rows)\n",
        "    validate_trajectory(context, rows[:1])\n",
    ),
    (
        "replay-separate-scorer-caches",
        "replay.py",
        "computation=shared_cache",
        "computation=CachedComputation()",
    ),
    (
        "candidate-untracked-cleanliness",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "    if status.strip():",
        "    if False:",
    ),
    (
        "candidate-wheel-committed-source",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "                if archive.read(member) != committed:",
        "                if False:",
    ),
    (
        "candidate-runtime-resource-recheck",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "resources = self.provenance.verify()",
        "resources = self.provenance.files",
    ),
    (
        "candidate-forbidden-import-call",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            validate_imports(receipt, **validation_policy)",
        "            pass",
    ),
    (
        "policy-package-authorization",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'identity["package_digest"] != self.package_digest',
        "False",
    ),
    (
        "policy-full-identity",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'result["identity"] != self._health_identity\n'
        '                or result["identity"] != self._latest_health_identity',
        "False",
    ),
    (
        "policy-output-classification",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'category == "PolicyOutputError"',
        "False",
    ),
    (
        "policy-aperture-classification",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'category == "ValueError" and message == "Negative gripper aperture"',
        "False",
    ),
    (
        "policy-nonfinite-target",
        "control.py",
        "if joints.shape != (prefix, 2, 7) or not np.isfinite(joints).all():",
        "if joints.shape != (prefix, 2, 7):",
    ),
    (
        "policy-public-command-replay",
        "policy_entry.py",
        'if request.get("command_replay", False):',
        "if False:",
    ),
    ("policy-ik-rejected-evidence", "replay.py", 'if hasattr(exc, "diagnostics"):', "if False:"),
    (
        "policy-mme-authorization",
        "policy.py",
        "    def authorize(self):\n        raise ValueError(",
        "    def authorize(self):\n        return {}\n        raise ValueError(",
    ),
    (
        "policy-no-reverse",
        "policy_ik.py",
        "np.dot(anchor[:, 0], rotation[:, 0]) >= 1 - 1e-6",
        "False",
    ),
    (
        "policy-no-retry",
        "policy_ik.py",
        "        return self.client.solve(poses, seed, dt_s=dt_s)",
        "        self.client.solve(poses, seed, dt_s=dt_s)\n"
        "        return self.client.solve(poses, seed, dt_s=dt_s)",
    ),
    (
        "entrypoint-exact-path",
        "source_receipt.py",
        'if path == Path(binding["path"]).resolve()',
        "if True",
    ),
    (
        "entrypoint-committed-bytes",
        "source_receipt.py",
        'and row.get("sha256") == expected["sha256"] == sha(path)',
        "and True",
    ),
    (
        "sdk-extension-namespace-roots",
        "../../../../../scripts/maintainer/candidate_replay.py",
        '        roots["omni"].extend(sdk_omni_roots(Path(sdk_file).resolve().parent))',
        "        pass",
    ),
    (
        "policy-parent-completion",
        "../../../../../scripts/maintainer/candidate_policy.py",
        "    return finish(request, result.returncode, diagnostic=args.diagnostic)",
        "    return result.returncode",
    ),
    (
        "disable-sdk-crash-upload",
        "app.py",
        '            "enable_crashreporter": False,',
        '            "enable_crashreporter": True,',
    ),
    (
        "policy-existing-output",
        "../../../../../scripts/maintainer/candidate_policy.py",
        'if Path(request["output"]).exists():',
        "if False:",
    ),
    (
        "policy-parent-exclusive",
        "../../../../../scripts/maintainer/candidate_policy.py",
        '        save_exclusive(\n            output / "diagnostic-result.json",',
        '        save(\n            output / "diagnostic-result.json",',
    ),
    (
        "policy-seed-exclusive",
        "policy_entry.py",
        '    save_exclusive(\n        run / "diagnostic-result.json"',
        '    __import__("tower_sim.bench.resources", fromlist=["save"]).save(\n'
        '        run / "diagnostic-result.json"',
    ),
    (
        "policy-diagnostic-health",
        "policy_entry.py",
        "            validate_diagnostic_health(health, package)",
        "            pass",
    ),
    (
        "policy-diagnostic-prompt",
        "policy_entry.py",
        'if request["prompt"] != benchmark_protocol["tasks"][request["task"]]["prompt"]:',
        "if False:",
    ),
    (
        "policy-formal-cli-disabled",
        "../../../../../scripts/maintainer/candidate_policy.py",
        "    if not args.diagnostic:",
        "    if False:",
    ),
    ("policy-rpc-events", "metrics.py", "rpc_observer=observe", "rpc_observer=None"),
    ("policy-nonjson-error-evidence", "policy.py", ' and getattr(exc, "body_is_json", True)', ""),
    (
        "entrypoint-startup-digest",
        "source_receipt.py",
        '            and row.get("startup_sha256") == expected["sha256"]\n',
        "",
    ),
    (
        "policy-history-rejected-callsite",
        "replay.py",
        "rejected=index in newly_rejected,",
        "rejected=False,",
    ),
    (
        "policy-history-terminal-callsite",
        "replay.py",
        "terminal=failures[index] is not None or terminal_ticks[index] == at_tick,",
        "terminal=False,",
    ),
    (
        "policy-formal-identity-verification",
        "replay.py",
        '                                    evaluation.verify_prediction(predicted["identity"])\n',
        "                                    pass\n",
    ),
    (
        "policy-start-failure-close",
        "replay.py",
        "            health = policy.start(context.output)\n"
        "        except BaseException:\n"
        "            policy.close()\n",
        "            health = policy.start(context.output)\n"
        "        except BaseException:\n"
        "            pass\n",
    ),
    (
        "policy-library-formal-disabled",
        "policy_entry.py",
        "    if not diagnostic and type(verification) is not ReleaseVerification:\n"
        "        raise ValueError(\n",
        "    if False:\n        raise ValueError(\n",
    ),
    ("policy-rpc-timer-boundary", "metrics.py", "        timer.close()\n", ""),
    (
        "desktop-admission-inventory",
        "../../../../../scripts/maintainer/gpu_admission.py",
        '                sample.update(desktop_sample(sample["xml"]))',
        "                sample.update(memory_mib=0, utilization_percent=0, processes=[])",
    ),
    (
        "desktop-profile-comparison-binding",
        "../../../../../scripts/maintainer/gpu_baseline.py",
        "    comparison_identity(left_run, right_run, informational=informational_cross_gpu)",
        "    pass",
    ),
    (
        "kit-crash-environment-disabled",
        "../../../../../scripts/maintainer/kit_safety.py",
        '        OMNI_CRASHREPORTER_ENABLED="0",\n',
        "",
    ),
    (
        "public-default-switch",
        "../../../../../tools/simulator/benchmark.py",
        "    return benchmark_candidate.command(args)",
        "    return 0",
    ),
    (
        "public-portable-switch",
        "../../../../../tools/simulator/benchmark_portable.py",
        "    return benchmark_candidate.portable(args)",
        "    return 0",
    ),
    (
        "public-diagnostic-switch",
        "../../../../../tools/simulator/benchmark_diagnostic.py",
        "    return benchmark_candidate.diagnostic(options)",
        "    return 0",
    ),
    (
        "public-runner-factory",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "        public=True, benchmark_version=benchmark_version, data_root=data_root, v1_root=v1_root",
        "        public=False, benchmark_version=benchmark_version, data_root=data_root, "
        "v1_root=v1_root",
    ),
    (
        "public-metrics-environment",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "if key not in METRICS_ENV",
        "if True",
    ),
    (
        "public-data-verification",
        "verification.py",
        "if digest(contents) != expected:",
        "if False:",
    ),
    (
        "public-package-version",
        "verification.py",
        '        if actual != expected:\n            raise ValueError(f"Unsupported package',
        '        if False:\n            raise ValueError(f"Unsupported package',
    ),
    (
        "public-release-package-closure",
        "verification.py",
        '    if set(body["packages"]) != set(PACKAGES):',
        "    if False:",
    ),
    (
        "public-release-wheel",
        "verification.py",
        'if digest(contents) != entry["sha256"]:',
        "if False:",
    ),
    (
        "public-installed-wheel",
        "verification.py",
        "            if loaded.read_bytes() != original:",
        "            if False:",
    ),
    (
        "public-dependency-manifest",
        "verification.py",
        '        if actual != expected:\n            raise ValueError("Installed dependencies',
        '        if False:\n            raise ValueError("Installed dependencies',
    ),
    (
        "public-formal-verification",
        "policy_entry.py",
        "        verification.verify()\n",
        "        pass\n",
    ),
    (
        "public-smoke-release",
        "../../../../../scripts/maintainer/public_smoke.py",
        'release=public.verify_release(benchmark.ROOT / "dist/release-manifest.json"),',
        "release={},",
    ),
    ("public-untracked-bound-trees", "verification.py", "        if extra:", "        if False:"),
    (
        "public-sdk-overlap-rejection",
        "verification.py",
        "if any(root.is_relative_to(tree) or tree.is_relative_to(root) for tree in trees):",
        "if False:",
    ),
    (
        "public-in-checkout-sdk-import",
        "verification.py",
        "elif any(path.is_relative_to(p) for p in roots):",
        "elif False:",
    ),
    (
        "public-bound-tree-before-sdk",
        "verification.py",
        "elif any(path.is_relative_to(p.resolve()) for p in guarded):",
        "elif False:",
    ),
    (
        "public-startup-import-closure",
        "verification.py",
        "    verify_sources(imported_sources(), release, allowed_roots=roots)",
        "    pass",
    ),
    (
        "public-imported-snapshot",
        "verification.py",
        "                files[path] = sha(path)",
        "                pass",
    ),
    (
        "public-preflight-startup-closure",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "    verify_startup(release)\n",
        "",
    ),
    ("public-formal-startup-closure", "verification.py", "        verify_startup(release)\n", ""),
    (
        "public-replay-identity-startup",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            verify_startup(release)\n",
        "",
    ),
    (
        "public-replay-setup-startup",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            verify_startup(self.release)\n",
        "            pass\n",
    ),
    (
        "public-smoke-startup",
        "../../../../../scripts/maintainer/public_smoke.py",
        '    public.verify_startup(identity["release"])\n',
        "",
    ),
    (
        "public-neutral-formal-message",
        "policy_entry.py",
        "Formal runs require a ReleaseVerification; use diagnostic=True otherwise.",
        "Formal candidate runs are enabled at P7-10 after G(all12) and S pass",
    ),
    (
        "public-worker-verifier-before-side-effects",
        "../../../../../scripts/maintainer/candidate_policy.py",
        "if verification is not None and type(verification) is not ReleaseVerification:",
        "if False:",
    ),
    (
        "public-portable-root-required-value",
        "verification.py",
        'if not value or value.startswith("--"):',
        "if False:",
    ),
    (
        "public-portable-root-equals",
        "verification.py",
        'argument.startswith("--portable-root=")',
        'argument.startswith("--not-portable-root=")',
    ),
    (
        "public-portable-root-option",
        "verification.py",
        "        generated = portable_root(sys.argv[1:])",
        "        generated = None",
    ),
    (
        "public-guide-layout-untracked-permission",
        "verification.py",
        '"--untracked-files=no"',
        '"--untracked-files=all"',
    ),
    (
        "public-startup-layout-recheck",
        "verification.py",
        '    verify_checkout_layout(Path(release["checkout"]))\n',
        "",
    ),
    (
        "public-release-builder-layout",
        "../../../../../scripts/maintainer/candidate_release.py",
        "    verify_checkout_layout(root)\n",
        "",
    ),
    (
        "public-importer-startup",
        "verification.py",
        "        source = module_source(name, module)",
        '        source = getattr(module, "__file__", None)\n'
        "        if source and not Path(source).is_file():\n"
        "            source = None",
    ),
    (
        "public-importer-complete",
        "source_receipt.py",
        "            source = module_source(name, module)",
        '            source = getattr(module, "__file__", None)\n'
        "            if source and not Path(source).is_file():\n"
        "                source = None",
    ),
    (
        "public-importer-rejection",
        "source_receipt.py",
        '    raise ValueError(f"Unbound non-file import: {name}: {source!r} ({type(loader).__name__})")',
        "    return None",
    ),
    (
        "public-importer-sdk-archive",
        "source_receipt.py",
        "archive.is_file() and any(archive.is_relative_to(root) for root in roots)",
        "archive.is_file()",
    ),
    (
        "purge-resource-default",
        "../backend/resources.py",
        "def default_path(name):\n    return DATA / PATHS[name]",
        "def default_path(name):\n    return Path(PATHS[name])",
    ),
    (
        "public-runtime-profile-canonical",
        "../backend/runtime_profiles.py",
        "from tower_sim.replay.dual_core import (",
        "from replay_runner_" + "isaac_sim.dual_core import (",
    ),
    ("public-wheel-yaml", "../../../pyproject.toml", '"pyyaml>=6,<7", ', ""),
    ("public-wheel-h5py", "../../../pyproject.toml", '"h5py>=3.11,<4", ', ""),
    (
        "public-bytecode-0",
        "../../../../../tools/simulator/benchmark_environment.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-1",
        "../../../../../tools/simulator/benchmark_diagnostic.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-2",
        "../../../../../tools/simulator/benchmark.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-3",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-4",
        "../../../../../tools/simulator/benchmark_portable.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-5",
        "../../../../../tools/simulator/benchmark_assets.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-6",
        "../../../../../tools/simulator/benchmark_diagnostic_score.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-7",
        "../../../../../scripts/maintainer/check_golden_exact.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-8",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-9",
        "../../../../../scripts/maintainer/gpu_baseline.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-10",
        "../../../../../scripts/maintainer/candidate_release.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-11",
        "../../../../../scripts/maintainer/candidate_policy.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-bytecode-12",
        "../../../../../scripts/maintainer/bench_scoring_replay.py",
        "sys.dont_write_bytecode = True",
        "sys.dont_write_bytecode = False",
    ),
    (
        "public-reference-help-benchmark_portable",
        "../../../../../tools/simulator/benchmark_portable.py",
        'help="Unavailable: the frozen reference runtime is maintained in the private archive"',
        'help="Maintainer frozen reference"',
    ),
    (
        "public-reference-help-benchmark_diagnostic",
        "../../../../../tools/simulator/benchmark_diagnostic.py",
        'help="Unavailable: the frozen reference runtime is maintained in the private archive"',
        'help="Maintainer frozen reference"',
    ),
    (
        "public-reference-help-benchmark_diagnostic_score",
        "../../../../../tools/simulator/benchmark_diagnostic_score.py",
        'help="Unavailable: the frozen reference runtime is maintained in the private archive"',
        'help="Maintainer frozen reference"',
    ),
    (
        "public-importer-native-root",
        "source_receipt.py",
        "                    if path.is_file() and any(path.is_relative_to(root) for root in roots):",
        "                    if path.is_file():",
    ),
    (
        "public-calibration-explicit-path",
        "environment.py",
        'calibration_report=resources.path("calibration_report"),',
        "",
    ),
    (
        "public-calibration-report-read",
        "../backend/runtime_profiles.py",
        "report = json.loads(calibration_report.read_text())",
        "report = json.loads(REPORT.read_text())",
    ),
    (
        "public-calibration-provenance",
        "../backend/runtime_profiles.py",
        '"reference_report": str(calibration_report)',
        '"reference_report": str(REPORT)',
    ),
    (
        "public-importer-none",
        "source_receipt.py",
        "if module is None:\n        # The import-blocking sentinel carries no executable payload.\n        return None",
        'if module is None:\n        raise ValueError("Unbound import-blocking sentinel")',
    ),
    (
        "public-importer-provider-namespace",
        "source_receipt.py",
        'type(loader).__module__.split(".")[0] == name.split(".")[0]',
        "True",
    ),
    (
        "public-importer-runtime-spec",
        "source_receipt.py",
        "if loader is None and spec is None:",
        "if loader is None:",
    ),
    ("public-wheel-pillow", "../../../pyproject.toml", '"pillow>=10,<13", ', ""),
    (
        "scene-v1-reader",
        "../../../../../scripts/maintainer/archive_scene.py",
        'result["candidate"] = relocate(result["candidate"])',
        'result["candidate"] = result["candidate"]',
    ),
    (
        "scene-map-check",
        "../../../../../scripts/maintainer/scene_v2_release.py",
        "elif (target / name).read_bytes() != data:",
        "elif False:",
    ),
    (
        "scene-leak-gate",
        "../../../../../tests/test_no_internal_leaks.py",
        "findings.extend(scan_path(root, name))",
        "pass",
    ),
    (
        "scene-redaction",
        "../../../../../scripts/maintainer/archive_scene.py",
        "parent.pop(parts[-1], None)",
        "pass",
    ),
    (
        "scene-envelope-version",
        "verification.py",
        'benchmark=benchmark,\n        data_version="benchmark-" + benchmark,',
        'benchmark="2026.11.0",\n        data_version="benchmark-2026.11.0",',
    ),
    (
        "scene-data-manifest",
        "data_release.py",
        'hashlib.sha256(payload).hexdigest() != expected["sha256"]',
        "False",
    ),
    (
        "scene-cross-revision-proof",
        "../../../../../scripts/maintainer/scene_equivalence.py",
        'raise ValueError(f"Scene equivalence mismatch: {key}")',
        "pass",
    ),
    (
        "scene-internal-lineage-boundary",
        "../../../../../scripts/maintainer/scene_v2_release.py",
        '        raise ValueError("Internal lineage must be outside the checkout and public release")',
        "        return {}",
    ),
    (
        "scene-full-runtime-equality",
        "../../../../../scripts/maintainer/archive_scene.py",
        "if expected != resolved:",
        "if False:",
    ),
    (
        "scene-measured-asset-retained",
        "../../../../../scripts/maintainer/dropped-fields.json",
        '"dropped": [',
        '"dropped": ["/candidate/episode_spec/blocks/*/measured_asset_id",',
    ),
    (
        "scene-dropped-read-enforcement",
        "../../../../../scripts/maintainer/archive_scene.py",
        "    return findings",
        "    return []",
    ),
    (
        "scene-construction-measured-id",
        "scene.py",
        '-1 if block["measured_asset_id"] is None else block["measured_asset_id"]',
        "-1",
    ),
    (
        "scene-asset-closure",
        "scene_format.py",
        '"asset_closure_verified": asset_manifest is not None',
        '"asset_closure_verified": True',
    ),
    (
        "scene-selector-range",
        "../../../../../scripts/maintainer/archive_scene.py",
        "0 <= selector < 50",
        "-1 <= selector <= 50",
    ),
    (
        "scene-selector-hash",
        "../../../../../scripts/maintainer/archive_scene.py",
        'r["old_sha256"] == digest(original)',
        "True",
    ),
    (
        "scene-archive-hash",
        "../../../../../scripts/maintainer/archive_scene.py",
        'digest(row) == entries[0]["old_sha256"]',
        "True",
    ),
    (
        "scene-invalid-archive-root",
        "../../../../../scripts/maintainer/archive_root.py",
        'if not (bundle / "protocol.json").is_file():',
        "if False:",
    ),
    ("scene-private-regex", "scene_format.py", "or PRIVATE_STRING.search(value)", "or False"),
    ("scene-canonical-resources", "scene_format.py", "if expected != body:", "if False:"),
    (
        "purge-archive-pins",
        "../../../../../scripts/maintainer/archive_scene.py",
        "if hashlib.sha256(payload).hexdigest() != expected:",
        "if False:",
    ),
    (
        "scene-lineage-placement",
        "public_surface.py",
        "if any(token in payload for token in schemas):",
        "if False:",
    ),
    (
        "scene-public-file-set",
        "../../../../../tests/test_no_internal_leaks.py",
        'if actual != set(manifest()["files"]):',
        "if False:",
    ),
    (
        "scene-archive-import-isolation",
        "scene_format.py",
        "def scene_schema():",
        "def scene_schema():\n    from . import archive_compat",
    ),
    (
        "scene-locator-independent-identity",
        "verification.py",
        "data_manifest_sha256=digest(data_manifest_path(benchmark).read_bytes()),",
        "data_manifest_sha256=digest(data_manifest_path(benchmark).read_bytes() + "
        "((data_manifest_path(benchmark).parents[5] / "
        '"benchmarks/2026.11.1/assets.lock.json").read_bytes() if benchmark == DEFAULT_BENCHMARK else '
        'b"")),',
    ),
    (
        "scene-locator-manifest",
        "data_release.py",
        'if lock["manifest_sha256"] != document["manifest_sha256"]:',
        "if False:",
    ),
    (
        "scene-schema-allowlist",
        "scene_format.py",
        "or set(value) - properties.keys():",
        "or False:",
    ),
    (
        "scene-default-revision",
        "../../../../../tools/simulator/benchmark_assets.py",
        "benchmark_version=DEFAULT_BENCHMARK,",
        'benchmark_version="2026.11.0",',
    ),
    (
        "scene-data-upload-manifest",
        "data_release.py",
        'if hashlib.sha256(payload).hexdigest() != document["manifest_sha256"]:',
        "if False:",
    ),
    (
        "public-importer-runtime-admission",
        "source_receipt.py",
        "if loader is None and spec is None:",
        "if False:",
    ),
    (
        "public-importer-runtime-relative",
        "source_receipt.py",
        "if loader is None and spec is None:",
        "if not source and loader is None and spec is None:",
    ),
    (
        "public-importer-runtime-loader",
        "source_receipt.py",
        "if loader is None and spec is None:",
        "if spec is None:",
    ),
    (
        "public-importer-provider-general",
        "source_receipt.py",
        'type(loader).__module__.split(".")[0] == name.split(".")[0]',
        'name.split(".")[0] in {"omni", "carb", "pxr", "isaacsim"} and type(loader).__module__.split(".")[0] == name.split(".")[0]',
    ),
    (
        "public-importer-provider-root",
        "source_receipt.py",
        "if path.is_file() and any(path.is_relative_to(root) for root in roots):\n            return path",
        "if path.is_file():\n            return path",
    ),
    (
        "public-importer-provider-file",
        "source_receipt.py",
        "if path.is_file() and any(path.is_relative_to(root) for root in roots):\n            return path",
        "if any(path.is_relative_to(root) for root in roots):\n            return path",
    ),
)


MUTATIONS += (
    (
        "scene-download-locator-directory",
        "data_release.py",
        'safe_path(Path.cwd(), lock["directory"].strip("/") or ".")',
        'safe_path(Path.cwd(), lock["directory"])',
    ),
    (
        "scene-download-directory-slashes",
        "data_release.py",
        '    directory = lock["directory"].strip("/")',
        '    directory = lock["directory"]',
    ),
    (
        "scene-download-root-prefix",
        "data_release.py",
        'prefix = "" if directory in ("", ".") else quote(directory) + "/"',
        'prefix = quote(directory) + "/"',
    ),
    (
        "scene-download-stale-temporaries",
        "data_release.py",
        "            temporary.unlink()",
        "            pass",
    ),
    (
        "scene-readme-pin-check",
        "../../../../../scripts/maintainer/scene_v2_release.py",
        "d.verify_locator(json.load(open('benchmarks/2026.11.1/assets.lock.json')), require_pin=True)",
        "d.verify_locator(json.load(open('benchmarks/2026.11.1/assets.lock.json')), require_pin=False)",
    ),
    (
        "scene-readme-download-check",
        "../../../../../scripts/maintainer/scene_v2_release.py",
        "d.download('.', scratch.name); ",
        "",
    ),
    (
        "scene-readme-verification-check",
        "../../../../../scripts/maintainer/scene_v2_release.py",
        "d.verify_release(d.payload_root(scratch.name))",
        "{}",
    ),
    (
        "scene-assets-verify-data-root",
        "../../../../../tools/simulator/benchmark_assets.py",
        '            result["data_files"] = len(data_release.verify_release(args.data_root))',
        "            pass",
    ),
    (
        "scene-runtime-labels-binding",
        "../../../../../scripts/maintainer/archive_scene.py",
        "    if unexpected:\n",
        "    if False:\n",
    ),
    (
        "scene-runtime-labels-equality",
        "../../../../../scripts/maintainer/archive_scene.py",
        "    assert_runtime_labels(\n"
        "        remove_declared_fields(relocate_v1(original, assets, asset_manifest)), expected\n"
        "    )\n",
        "",
    ),
    (
        "scene-empty-manifest-rejection",
        "scene_format.py",
        "        if asset_manifest is not None\n",
        "        if asset_manifest\n",
    ),
    (
        "scene-neutral-task7-frame",
        "../../../../../scripts/maintainer/archive_scene.py",
        '        result["initial_state"]["coordinate_frame"] = "encoder_with_single_J2_conversion"\n',
        "",
    ),
    (
        "scene-task-scene-message",
        "data_release.py",
        "Expected exactly one scene task/scene",
        "Expected exactly one scene task/seed",
    ),
    (
        "purge-tree-generation-gate",
        "public_surface.py",
        "    check_generation(name, payload)",
        "    pass",
    ),
    (
        "purge-tree-archive-gate",
        "public_surface.py",
        "    check_archive(name, payload)",
        "    pass",
    ),
    ("purge-leak-gate", "public_surface.py", "    if findings:\n", "    if False:\n"),
    (
        "purge-generation-module",
        "public_surface.py",
        '    if "tower-" + "datagenerator" in parts or any(token in dotted for token in '
        "REMOVED_MODULES):",
        "    if False:",
    ),
    (
        "purge-generation-symbol",
        "public_surface.py",
        'n in REMOVED_SYMBOLS or any(n == m or n.startswith(m + ".") for m in REMOVED_MODULES)',
        "False",
    ),
    (
        "purge-archive-path",
        "public_surface.py",
        "    if token in PurePosixPath(name).parts or PurePosixPath(name).stem == token:",
        "    if False:",
    ),
    (
        "purge-archive-import",
        "public_surface.py",
        '            if any(token in n.split(".") for n in names):',
        "            if False:",
    ),
    (
        "purge-leak-storage-host",
        "public_surface.py",
        "    findings = [match.group() for match in FORBIDDEN.finditer(line)]",
        "    findings = []",
    ),
    ("purge-leak-ip", "public_surface.py", "        findings.append(str(address))", "        pass"),
    (
        "purge-leak-email",
        "public_surface.py",
        "    findings.extend(match.group() for match in EMAIL.finditer(line))",
        "    pass",
    ),
    (
        "purge-allowlist",
        "../../../../../tests/test_no_internal_leaks.py",
        "FROZEN_ALLOWLIST = {}",
        'FROZEN_ALLOWLIST = {"*": {"reason": "restored obsolete exception"}}',
    ),
    (
        "purge-wheel-release-scan",
        "verification.py",
        "            check_public(member, archive.read(member))",
        "            pass",
    ),
    (
        "purge-checkout-release-scan",
        "verification.py",
        "            check_public(name, path.read_bytes())",
        "            pass",
    ),
    (
        "purge-checkout-tree-scan",
        "../../../../../scripts/maintainer/check_public_tree.py",
        "            check_public(name, path.read_bytes())",
        "            pass",
    ),
    (
        "purge-dropped-generic-key",
        "../../../../../scripts/maintainer/archive_scene.py",
        "                    and key.value in keys",
        '                    and key.value in keys and ast.unparse(base) == "candidate"',
    ),
    (
        "purge-package-hash",
        "verification.py",
        "        if sha(path) != expected:\n"
        '            raise ValueError(f"Changed evaluation package data: {name}")',
        '        if False:\n            raise ValueError(f"Changed evaluation package data: {name}")',
    ),
    (
        "purge-package-release-binding",
        "verification.py",
        '        data["files"].update(verify_package_data())',
        "        pass",
    ),
    (
        "purge-release-trees",
        "verification.py",
        'CHECKOUT_TREES = ("tools/simulator", "scripts")',
        'CHECKOUT_TREES = ("tools/simulator",)',
    ),
    (
        "purge-archive-explicit-root",
        "../../../../../scripts/maintainer/archive_root.py",
        "    if selected is None:",
        "    if False:",
    ),
    (
        "purge-golden-reader",
        "../../../../../scripts/maintainer/golden_support.py",
        "    return archive_bundle(reference_root)",
        '    return Path.cwd() / "archive"',
    ),
    (
        "purge-replay-output",
        "../replay/replay_actions.py",
        '    raise ValueError("Standalone replay requires an explicit output directory")',
        "    return DEFAULT_OUTPUT",
    ),
    (
        "purge-exact-no-archive",
        "../../../../../scripts/maintainer/check_golden_exact.py",
        "dict(passed=False, skipped=True, reason=str(error))",
        "dict(passed=True, skipped=True, reason=str(error))",
    ),
    (
        "purge-bench-exact-no-archive",
        "../../../../../scripts/maintainer/bench_scoring_replay.py",
        "dict(passed=False, skipped=True, reason=str(error))",
        "dict(passed=True, skipped=True, reason=str(error))",
    ),
    (
        "purge-reader-tasks",
        "resources.py",
        'tasks=DATA / "registry/tasks.json"',
        'tasks=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-asset_lock",
        "resources.py",
        'asset_lock=DATA / "benchmark/assets.lock.json"',
        'asset_lock=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-scene_cache",
        "resources.py",
        'scene_cache=capsule / "runtime/scene-cache.json"',
        'scene_cache=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-runtime_profiles",
        "resources.py",
        'runtime_profiles=capsule / "runtime-profiles"',
        'runtime_profiles=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-shared_ik_spec",
        "resources.py",
        'shared_ik_spec=runtime / "shared/shared-ik-spec.json"',
        'shared_ik_spec=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-camera_profile",
        "resources.py",
        'camera_profile=runtime / "shared/camera-profile-near10.json"',
        'camera_profile=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-motion_profile",
        "resources.py",
        'motion_profile=resource_path(checkout, "motion_profile")',
        'motion_profile=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-calibration_report",
        "resources.py",
        'calibration_report=resource_path(checkout, "calibration_report")',
        'calibration_report=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-tcp_cache",
        "resources.py",
        'tcp_cache=DATA / "calibration/tcp-v2.json"',
        'tcp_cache=checkout / "removed-resource.json"',
    ),
    (
        "purge-reader-tcp_fit_receipt",
        "resources.py",
        'tcp_fit_receipt=DATA / "calibration/fit-equivalence.json"',
        'tcp_fit_receipt=checkout / "removed-resource.json"',
    ),
    (
        "purge-default-motion_profile",
        "../backend/resources.py",
        '"motion_profile": "motion_profile_v1.json"',
        '"motion_profile": "removed-resource.json"',
    ),
    (
        "purge-default-runtime_profiles",
        "../backend/resources.py",
        '"runtime_profiles": "runtime_profiles_20260902"',
        '"runtime_profiles": "removed-resource.json"',
    ),
    (
        "purge-default-calibration_report",
        "../backend/resources.py",
        '"calibration_report": "calibration/robot_dynamics.json"',
        '"calibration_report": "removed-resource.json"',
    ),
)

MUTATIONS += (
    (
        "purge-reader-archive-identity",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "    archive = bundle.parents[1]",
        "    archive = root",
    ),
    (
        "purge-reader-module-stems",
        "../../../../../scripts/maintainer/candidate_replay.py",
        '(archive_bundle(v1_root) / "SOURCE_MANIFEST.json").read_text()',
        '(root / "SOURCE_MANIFEST.json").read_text()',
    ),
    (
        "purge-reader-protocol",
        "../../../../../scripts/maintainer/candidate_replay.py",
        '(archive_bundle(v1_root) / "protocol.json").read_text()',
        '(root / "protocol.json").read_text()',
    ),
)

MUTATIONS += (
    (
        "purge-retired-bundle",
        "public_surface.py",
        '    if PurePosixPath(name).parts[:2] == ("simulator", "benchmark-202611"):',
        "    if False:",
    ),
)

MUTATIONS += (
    (
        "purge-reader-forbidden-archive",
        "../../../../../scripts/maintainer/candidate_replay.py",
        'forbidden_roots = [str(archive_bundle(self.provenance.v1_root) / "runtime")]',
        "forbidden_roots = []",
    ),
    (
        "purge-reader-portable-archive",
        "../../../../../tools/simulator/benchmark_portable.py",
        "        benchmark.refuse_reference()",
        "        return 0",
    ),
    (
        "purge-reader-tcp-archive",
        "../../../../../scripts/maintainer/check_tcp_fit.py",
        "archived_v1 = archive_checkout(args.archive_root) / V1",
        "archived_v1 = ROOT / V1",
    ),
    (
        "purge-reader-exact-corpus",
        "../../../../../scripts/maintainer/check_golden_exact.py",
        "args.fixture_root = args.fixture_root or golden_root(args.reference_root)",
        "args.fixture_root = args.fixture_root or GOLDEN",
    ),
    (
        "purge-reader-bench-corpus",
        "../../../../../scripts/maintainer/bench_scoring_replay.py",
        "golden = golden_root(args.archive_root)",
        'golden = ROOT / "tests/golden/benchmark-2026.11"',
    ),
)

MUTATIONS += (
    (
        "purge-golden-recording-boundary",
        "public_surface.py",
        'if name.startswith("tests/golden/") and name.endswith(".tar.xz"):',
        "if False:",
    ),
)

MUTATIONS += (
    ("purge-redaction-keys", "redaction.py", "if left.keys() != right.keys():", "if False:"),
    ("purge-redaction-length", "redaction.py", "if len(left) != len(right):", "if False:"),
    ("purge-redaction-leaf", "redaction.py", "elif not left == right:", "elif False:"),
    (
        "purge-redaction-stage",
        "drives.py",
        "            matches_exact(",
        "            (lambda *args: True)(",
    ),
)

MUTATIONS += (
    (
        "purge-ik-redacted-pin",
        "ik_identity.py",
        "if hashlib.sha256(path.read_bytes()).hexdigest() != REDACTED_SHARED_SPEC_SHA256:",
        "if False:",
    ),
    (
        "purge-ik-canonical-identity",
        "ik_identity.py",
        'return load_profile("command")["source_sha256"]',
        "return REDACTED_SHARED_SPEC_SHA256",
    ),
    (
        "public-ik-preflight-identity",
        "../../../../../tools/simulator/benchmark_candidate.py",
        '    spec_identity = shared_spec_identity()\n    if sha(request["ik_bounds"])',
        '    spec_identity = sha(spec)\n    if sha(request["ik_bounds"])',
    ),
    (
        "public-ik-prepare-identity",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "    spec_identity = shared_spec_identity()\n    spec = Resources",
        '    spec_identity = "invalid"\n    spec = Resources',
    ),
    (
        "public-ik-protocol-identity",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "ik=dict(shared_spec_sha256=spec_identity)",
        "ik=dict(shared_spec_sha256=sha(spec))",
    ),
    (
        "public-ik-review-identity",
        "../../../../../tools/simulator/benchmark_candidate.py",
        "!= spec_identity\n",
        "!= sha(spec)\n",
    ),
    (
        "policy-ik-canonical-review",
        "evaluation.py",
        "            == shared_spec_identity()\n",
        "",
    ),
)

MUTATIONS += (
    (
        "purge-archive-marker",
        "../../../../../conftest.py",
        '        and item.get_closest_marker("archive")\n',
        "",
    ),
    (
        "purge-archive-private-failure",
        "../../../../../conftest.py",
        'explicit = item.config.getoption("--v1-root") or os.environ.get("TOWER_ARCHIVE_ROOT")',
        "explicit = None",
    ),
    (
        "purge-archive-skip-location",
        "../../../../../conftest.py",
        'and not explicit\n    ):\n        report.outcome = "skipped"\n        report.longrepr = (str(item.path), item.location[1] + 1, str(call.excinfo.value))',
        'and not explicit\n    ):\n        report.outcome = "skipped"\n        report.longrepr = (str(item.path), 1, str(call.excinfo.value))',
    ),
)

MUTATIONS += (
    (
        "config-retained-tasks-guard",
        "../backend/tasks.py",
        "return float(min(delta.magnitude(), flipped.magnitude()))",
        "return float(delta.magnitude())",
    ),
    (
        "backend-profile-reader-guard",
        "../backend/profiles.py",
        "if digest(profile) != expected:",
        "if False:",
    ),
    (
        "backend-environment-class-guard",
        "../backend/backend.py",
        "class IsaacEnvironment:",
        "class IsaacEnvironment(object):",
    ),
)

MUTATIONS += tuple(
    (
        "purge-history-pattern-" + str(index),
        "public_surface.py",
        pattern,
        "never-match-retired-label",
    )
    for index, pattern in enumerate(
        (
            "paper-isaac5[1]",
            "refs/ta[g]s/",
            "tag benchmark-2026[.]11[.]0",
            "benchmark-2026[.]11[.]0:",
            "jenga[-]desktop",
        )
    )
)

MUTATIONS += (
    (
        "purge-provenance-reader",
        "j2comp.py",
        "        self.CONFIG = copy.deepcopy(configuration)",
        '        self.CONFIG = copy.deepcopy(configuration)\n        self.CONFIG["left_wrist_refinement"]["source_episode"]',
    ),
    (
        "purge-remove-declared-fields",
        "../../../../../scripts/maintainer/archive_scene.py",
        "            parent.pop(parts[-1], None)",
        "            pass",
    ),
    (
        "purge-exact-skip-exit",
        "../../../../../scripts/maintainer/check_golden_exact.py",
        "return 0 if args.allow_skip else 77",
        "return 0",
    ),
    (
        "purge-replay-skip-exit",
        "../../../../../scripts/maintainer/bench_scoring_replay.py",
        "return 0 if args.allow_skip else 77",
        "return 0",
    ),
    (
        "purge-g-revision",
        "../../../../../scripts/maintainer/gpu_baseline.py",
        'if getattr(args, "benchmark", "2026.11.1") != "2026.11.1":',
        "if False:",
    ),
)

MUTATIONS += (
    (
        "purge-tcp-skip-exit",
        "../../../../../scripts/maintainer/check_tcp_fit.py",
        'return 77 if receipt.get("skipped") and not args.allow_skip else 0',
        "return 0",
    ),
    (
        "purge-scene-skip-exit",
        "../../../../../scripts/maintainer/scene_v2_release.py",
        'return 77 if result.get("skipped") and not args.allow_skip else 0',
        "return 0",
    ),
    (
        "purge-bound-trees",
        "verification.py",
        'BOUND_TREES = (*CHECKOUT_TREES, "packages", "src")',
        'BOUND_TREES = (*CHECKOUT_TREES, "packages", "src", "simulator")',
    ),
    (
        "purge-summary-revision",
        "../../../../../scripts/maintainer/gpu_host_profile.py",
        '        document["benchmark_version"] = original.benchmark_version(identity)',
        "        pass",
    ),
    (
        "desktop-admission-schema",
        "../../../../../scripts/maintainer/gpu_admission.py",
        'schema="tower-desktop-gpu-admission-v2"',
        'schema="obsolete-admission-v1"',
    ),
)
MUTATIONS += tuple(
    (
        "purge-data-copy-" + str(index),
        "../../../../../benchmarks/2026.11/" + name,
        '{\n  "',
        '{\n   "',
    )
    for index, name in enumerate(
        ("tasks.json", "calibration/tcp-v2.json", "calibration/fit-equivalence.json")
    )
)

MUTATIONS += (
    (
        "purge-private-archive-skip",
        "../../../../../conftest.py",
        'if explicit and report.skipped and "SKIP: supply --archive-root" in str(report.longrepr):',
        "if False:",
    ),
)

# Bind each scene mutation to its intended regression, avoiding unrelated failures.

MUTATIONS += (
    (
        "public-assets-package-lock",
        "../../../../../tools/simulator/benchmark_assets.py",
        'LOCK = DATA / "benchmark/assets.lock.json"',
        'from scripts.maintainer.archive_root import ArchivedPath\nLOCK = ArchivedPath() / "assets.lock.json"',
    ),
    (
        "public-assets-scene-index",
        "../../../../../tools/simulator/benchmark_assets.py",
        'if body["scene_index_sha256"] != SCENE_INDEX_SHA256:',
        'from scripts.maintainer.archive_root import ArchivedPath\n    if body["scene_index_sha256"] != benchmark.sha256(ArchivedPath() / "scene-index.json"):',
    ),
    (
        "public-assets-scene-cache",
        "../../../../../tools/simulator/benchmark_assets.py",
        'DATA / "benchmark/runtime/ik-capsule/runtime/scene-cache.json"',
        '__import__("scripts.maintainer.archive_root", fromlist=["archive_bundle"]).archive_bundle() / "runtime/ik-capsule/runtime/scene-cache.json"',
    ),
    (
        "public-ik-protocol-rejection",
        "../../../../../tools/simulator/benchmark_candidate.py",
        '        if settings["ik"]["shared_spec_sha256"] != spec_identity:\n            raise ValueError("Protocol changes the canonical shared IK identity")',
        "",
    ),
    (
        "purge-public-archive-path-gate",
        "public_surface.py",
        "    check_public_archive_paths(name, payload)",
        "    pass",
    ),
    (
        "purge-required-benchmark-version",
        "../../../../../scripts/maintainer/gpu_identity.py",
        '        if value not in ("2026.11.0", "2026.11.1"):',
        "        if False:",
    ),
)
MUTATIONS += tuple(
    ("purge-tag-form-" + str(index), "public_surface.py", pattern, "never-match-retired-tag")
    for index, pattern in enumerate(
        (
            "datagen-archive-v[1]",
            "simulator-assets-v[12]",
            "simulator-v2[.]0[.]1",
            "git checkout benchmark-2026[.]11[.]0",
        )
    )
)


MUTATIONS += (
    (
        "maintainer-explicit-archive-disjunct",
        "../../../../../scripts/maintainer/candidate_replay.py",
        'if self.v1_root is not None or self.benchmark_version == "2026.11.0":',
        'if self.benchmark_version == "2026.11.0":',
    ),
    (
        "maintainer-explicit-archive-conjunction",
        "../../../../../scripts/maintainer/candidate_replay.py",
        'if self.v1_root is not None or self.benchmark_version == "2026.11.0":',
        'if self.v1_root is not None and self.benchmark_version == "2026.11.0":',
    ),
    (
        "purge-public-worker-archive-plant",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            forbidden_roots = []",
        "            archive_bundle()\n            forbidden_roots = []",
    ),
    (
        "public-worker-archive-regression",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            asset_identity(self.assets, files=files)",
        "            data_identity(self.root, self.assets, files=files, v1_root=self.v1_root)",
    ),
    (
        "public-worker-pinned-stem-guard",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "            forbidden_stems = matched_archive_stems(receipt)",
        "            forbidden_stems = set()",
    ),
    (
        "public-worker-resource-recheck",
        "../../../../../scripts/maintainer/candidate_replay.py",
        "        self.bind_data(current)",
        "        pass",
    ),
    (
        "public-assets-index-rejection",
        "../../../../../tools/simulator/benchmark_assets.py",
        '        raise ValueError("Asset release belongs to a different frozen scene suite")',
        "        pass",
    ),
    (
        "public-assets-station-rejection",
        "../../../../../tools/simulator/benchmark_assets.py",
        '        raise ValueError("Station differs from the campaign calibration")',
        "        pass",
    ),
    (
        "public-scenes-default-routing",
        "../../../../../tools/simulator/benchmark_assets.py",
        "    destination,\n    *,\n    reference=False,",
        "    destination,\n    *,\n    reference=True,",
    ),
    (
        "public-scenes-required-version",
        "../../../../../tools/simulator/benchmark_assets.py",
        '    if "benchmark" not in body:',
        "    if False:",
    ),
    (
        "public-request-required-version",
        "../../../../../tools/simulator/benchmark_candidate.py",
        '    if "benchmark" not in body:',
        "    if False:",
    ),
)


SCENE_TESTS = {
    "scene-download-locator-directory": "test_data_download_is_pinned_and_checks_every_byte",
    "scene-v1-reader": "test_v1_reader_preserves_every_original_field",
    "scene-map-check": "test_hash_map_detects_tampering",
    "scene-leak-gate": "test_planted_leak_in_bound_metadata_is_rejected",
    "scene-redaction": "test_remove_declared_fields_preserves_every_other_value",
    "scene-envelope-version": "test_envelope_revision_switch",
    "scene-data-manifest": "test_payload_manifest_binding",
    "scene-cross-revision-proof": "test_physical_map_hash_must_match_payload",
    "scene-internal-lineage-boundary": "test_internal_lineage_cannot_enter_public_stage",
    "scene-full-runtime-equality": "test_redaction_and_task7_physical_binding",
    "scene-measured-asset-retained": "test_dropped_fields_have_no_runtime_readers",
    "scene-dropped-read-enforcement": "test_dropped_fields_have_no_runtime_readers",
    "scene-construction-measured-id": "test_v2_runtime_plans_and_clone_authoring",
    "scene-asset-closure": "test_v2_runtime_plans_and_clone_authoring",
    "scene-selector-range": "test_archive_selector_bounds_and_hashes",
    "scene-selector-hash": "test_archive_selector_bounds_and_hashes",
    "scene-archive-hash": "test_archive_selector_bounds_and_hashes",
    "scene-invalid-archive-root": "test_recheck_skips_only_missing_implicit_evidence",
    "scene-private-regex": "test_private_string_validation",
    "scene-canonical-resources": "test_canonical_resource_and_unknown_field_rejection",
    "scene-lineage-placement": "test_wheel_rejects_private_lineage",
    "scene-public-file-set": "test_metadata_file_set_is_exact",
    "scene-archive-import-isolation": "test_public_reader_is_independent_of_archive",
    "scene-locator-independent-identity": "test_locator_independent_identity",
    "scene-locator-manifest": "test_locator_independent_identity",
    "scene-schema-allowlist": "test_canonical_resource_and_unknown_field_rejection",
    "scene-default-revision": "test_v2_runtime_plans_and_clone_authoring",
    "scene-data-upload-manifest": "test_upload_manifest_byte_pin",
    "scene-download-directory-slashes": "test_data_download_is_pinned_and_checks_every_byte",
    "scene-download-root-prefix": "test_data_download_is_pinned_and_checks_every_byte",
    "scene-download-stale-temporaries": "test_download_retry_removes_stale_temporary_files",
    "scene-readme-pin-check": "test_readme_post_upload_commands_validate_pin_and_download",
    "scene-readme-download-check": "test_readme_post_upload_commands_validate_pin_and_download",
    "scene-readme-verification-check": "test_readme_post_upload_commands_validate_pin_and_download",
    "scene-assets-verify-data-root": "test_assets_verify_cli_checks_explicit_data_root",
    "scene-runtime-labels-binding": "test_runtime_labels_reject_undeclared_changes",
    "scene-runtime-labels-equality": "test_runtime_labels_reject_undeclared_changes",
    "scene-empty-manifest-rejection": "test_empty_asset_manifest_cannot_authorize_scene",
    "scene-neutral-task7-frame": "test_neutral_task7_frame_label_and_archive_cleanup",
    "scene-task-scene-message": "test_missing_public_scene_names_task_scene",
}


MUTATIONS += (
    (
        "contrib-golden-skip-marker",
        "../../../../../conftest.py",
        'and item.get_closest_marker("golden")',
        "and True",
    ),
    (
        "contrib-assets-skip-marker",
        "../../../../../conftest.py",
        'and item.get_closest_marker("public_assets")',
        "and True",
    ),
    (
        "contrib-golden-hash",
        "../../../../../scripts/maintainer/fetch_golden.py",
        'or hashlib.sha256(payload).hexdigest() != expected["sha256"]',
        "or False",
    ),
    (
        "contrib-golden-size",
        "../../../../../scripts/maintainer/fetch_golden.py",
        'len(payload) != expected["size"]',
        "False",
    ),
    (
        "contrib-golden-offline",
        "../../../../../scripts/maintainer/fetch_golden.py",
        'if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TOWER_OFFLINE") == "1":',
        "if False:",
    ),
    (
        "contrib-golden-cache",
        "../../../../../scripts/maintainer/fetch_golden.py",
        "return verify(target, expected)",
        "return target",
    ),
    (
        "contrib-station-pin",
        "asset_identity.py",
        "if digest == PUBLIC_STATION_SHA256 else digest",
        "if True else digest",
    ),
    (
        "contrib-assets-pin",
        "asset_identity.py",
        "if digest == PUBLIC_ASSET_MANIFEST_SHA256 else digest",
        "if True else digest",
    ),
    (
        "contrib-scan-personal-token",
        "../../../../../scripts/maintainer/scan_public_assets.py",
        'if hashlib.sha256(token.lower().encode("utf-8")).hexdigest() in PERSONAL_TOKEN_SHA256:',
        "if False:",
    ),
    (
        "contrib-scan-usd",
        "../../../../../scripts/maintainer/scan_public_assets.py",
        "numeric_ranges = self.usd(payload, suffix, file, member)",
        "return",
    ),
    (
        "contrib-scan-images",
        "../../../../../scripts/maintainer/scan_public_assets.py",
        "with Image.open(stream) as picture:",
        "return\n            with Image.open(stream) as picture:",
    ),
    (
        "contrib-scan-nested",
        "../../../../../scripts/maintainer/scan_public_assets.py",
        "for item in archive.infolist():",
        "for item in []:",
    ),
    (
        "contrib-scan-errors",
        "../../../../../scripts/maintainer/scan_public_assets.py",
        "passed=not self.hits and not self.errors,",
        "passed=not self.hits,",
    ),
)


MUTATIONS += (
    (
        "contrib-tree-scanner-call",
        "../../../../../scripts/maintainer/check_public_tree.py",
        "scanner.scan(path, name)",
        "pass",
    ),
    (
        "contrib-wheel-scanner-call",
        "../../../../../scripts/maintainer/check_public_tree.py",
        "scanner.scan(path)",
        "pass",
    ),
    ("contrib-download-attempts", "download.py", "ATTEMPTS = 5", "ATTEMPTS = 1"),
    (
        "contrib-download-4xx",
        "download.py",
        "if isinstance(error, HTTPError) and 400 <= error.code < 500:",
        "if False:",
    ),
    (
        "contrib-download-stream-retry",
        "download.py",
        "except RETRY_ERRORS as error:",
        "except HTTPError as error:",
    ),
    (
        "contrib-download-cache-hash",
        "download.py",
        "and hashlib.sha256(target.read_bytes()).hexdigest() == expected",
        "and True",
    ),
    (
        "contrib-download-cache-size",
        "download.py",
        "and (size is None or target.stat().st_size == size)",
        "and True",
    ),
    ("contrib-download-payload-hash", "download.py", "digest.hexdigest() != expected", "False"),
    (
        "contrib-download-payload-size",
        "download.py",
        "(size is not None and count != size)",
        "False",
    ),
    (
        "contrib-assets-partial-resume",
        "../../../../../scripts/maintainer/fetch_public_test_data.py",
        '".partial-" + lock["manifest_sha256"]',
        '".fresh-" + lock["manifest_sha256"]',
    ),
    (
        "contrib-assets-lock-error",
        "../../../../../scripts/maintainer/fetch_public_test_data.py",
        'raise ValueError("Public assets must use an immutable HF commit")',
        'raise PublicAssetsUnavailable("Public assets must use an immutable HF commit")',
    ),
    (
        "contrib-golden-lock-error",
        "../../../../../scripts/maintainer/fetch_golden.py",
        'raise ValueError("Public golden release must use an immutable HF commit")',
        'raise GoldenUnavailable("Public golden release must use an immutable HF commit")',
    ),
    (
        "contrib-replay-station-identity",
        "../replay/dual_replay_worker.py",
        "stage_identity_before = station_identity(stage_hash_before)",
        "stage_identity_before = stage_hash_before",
    ),
    (
        "contrib-replay-unknown-stage",
        "../replay/dual_replay_worker.py",
        "profile_source_stage_verified = provenance_hash == stage_identity_before",
        "profile_source_stage_verified = True",
    ),
    (
        "contrib-required-golden",
        "../../../../../conftest.py",
        'and item.get_closest_marker("golden")\n        and os.environ.get("TOWER_REQUIRE_PUBLIC_DATA") != "1"',
        'and item.get_closest_marker("golden")\n        and True',
    ),
    (
        "contrib-required-public_assets",
        "../../../../../conftest.py",
        'and item.get_closest_marker("public_assets")\n        and os.environ.get("TOWER_REQUIRE_PUBLIC_DATA") != "1"',
        'and item.get_closest_marker("public_assets")\n        and True',
    ),
)


MUTATIONS += (
    (
        "contrib-replay-alignment-identity",
        "../replay/dual_replay_worker.py",
        "runtime_stage_sha256=stage_identity_before if reference_report",
        "runtime_stage_sha256=stage_hash_before if reference_report",
    ),
    (
        "contrib-archive-assets-hash",
        "../../../../../scripts/maintainer/archive_assets.py",
        "hashlib.sha256(payload).hexdigest() != expected",
        "False",
    ),
    (
        "contrib-archive-assets-size",
        "../../../../../scripts/maintainer/archive_assets.py",
        "size is not None and len(payload) != size",
        "False",
    ),
    (
        "contrib-workflow-permissions",
        "../../../../../.github/workflows/ci.yml",
        "permissions:\n  contents: read\n",
        "",
    ),
    (
        "contrib-workflow-gpu-runner",
        "../../../../../.github/workflows/ci.yml",
        "quality:\n    runs-on: ubuntu-latest",
        "quality:\n    runs-on: [self-hosted, tower-gpu]",
    ),
    (
        "contrib-workflow-public-data",
        "../../../../../.github/workflows/ci.yml",
        'TOWER_REQUIRE_PUBLIC_DATA: "1"',
        'TOWER_REQUIRE_PUBLIC_DATA: "0"',
    ),
)


MUTATIONS += (
    (
        "contrib-archive-destination-checkout",
        "../../../../../scripts/maintainer/archive_assets.py",
        "if destination.is_relative_to(checkout):",
        "if False:",
    ),
    (
        "contrib-download-clean-truncation",
        "download.py",
        "if expected_length is not None and count < expected_length:",
        "if False:",
    ),
    (
        "contrib-assets-flock-X1",
        "../../../../../scripts/maintainer/fetch_public_test_data.py",
        "fcntl.flock(guard, fcntl.LOCK_EX)",
        "pass",
    ),
    (
        "contrib-assets-published-X2",
        "../../../../../scripts/maintainer/fetch_public_test_data.py",
        "                if not manifest.is_file():",
        "                if True:",
    ),
    ("contrib-download-5xx-X6", "download.py", "400 <= error.code < 500", "400 <= error.code"),
    (
        "contrib-replay-exact-identity-X11",
        "../replay/dual_replay_worker.py",
        "profile_source_stage_verified = provenance_hash == stage_identity_before",
        "profile_source_stage_verified = provenance_hash in (stage_identity_before, stage_hash_before)",
    ),
    (
        "contrib-assets-stale-temporaries",
        "../../../../../scripts/maintainer/fetch_public_test_data.py",
        "                        temporary.unlink()",
        "                        pass",
    ),
    (
        "contrib-assets-exact-file-set",
        "../../../../../scripts/maintainer/fetch_public_test_data.py",
        "if actual != expected:",
        "if False:",
    ),
    (
        "contrib-assets-empty-published",
        "../../../../../scripts/maintainer/fetch_public_test_data.py",
        "                if not manifest.is_file():",
        "                if not selected.is_dir():",
    ),
    (
        "contrib-workflow-custom-runner",
        "../../../../../.github/workflows/ci.yml",
        "quality:\n    runs-on: ubuntu-latest",
        "quality:\n    runs-on: tower-gpu",
    ),
)


MUTATIONS += (
    (
        "policy-env-order",
        "policy.py",
        "for index in sorted(payloads):",
        "for index in sorted(payloads, reverse=True):",
    ),
    (
        "policy-repr-identity",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'result["identity"] != self._health_identity',
        'repr(result["identity"]) != repr(self._health_identity)',
    ),
    (
        "policy-aperture-prefix",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'message == "Negative gripper aperture"',
        'message.startswith("Negative gripper aperture")',
    ),
    (
        "policy-error-code",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'if code in {"call_failed", "infer_failed"}:',
        "if True:",
    ),
)


MUTATIONS += (
    (
        "policy-timeout",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        "timeout_s=300.0",
        "timeout_s=120.0",
    ),
    (
        "policy-missing-prediction",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'elif getattr(exc, "details", {}).get("exception_type") == "KeyError":',
        "elif False:",
    ),
    (
        "policy-row-metadata",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        "if key in row and row[key] != full[key]:",
        "if False:",
    ),
    (
        "policy-row-identity",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'if row.get("identity") != result["identity"]:',
        "if False:",
    ),
    (
        "policy-png-format",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'if image.format != "PNG":',
        "if False:",
    ),
    (
        "policy-transport-label",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'transport="websocket"',
        'transport="unknown"',
    ),
    (
        "policy-aperture-type",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'category == "ValueError" and message',
        "message",
    ),
    (
        "policy-identity-normalize",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        'normalize(self.bridge._call("get_identity"))',
        'self.bridge._call("get_identity")',
    ),
    (
        "policy-atomic-infer",
        "../../../../../src/tower-eval/src/tower_eval/policies/runtime_ws.py",
        "rows = self.bridge.infer(mapped)",
        'self.bridge._call("update_obs", mapped)\n                    rows = self.bridge._call("get_action")',
    ),
)


MUTATIONS += (
    (
        "policy-adapter-infer-wire",
        "../../../../../src/tower-eval/src/tower_eval/policies/xpolicylab.py",
        '        with contextlib.redirect_stdout(sys.stderr):\n            client = self._client\n            response = client._run(\n                client._client.infer(\n                    observation,\n                    trial_id=client.trial_id,\n                    action_case_id=client.action_case_id,\n                    repeat_index=client.repeat_index,\n                    step=client._step,\n                )\n            )\n            client._step += 1\n            return response.payload["actions"]\n',
        '        self._call("update_obs", observation)\n        return self._call("get_action")\n',
    ),
    (
        "policy-adapter-infer-step",
        "../../../../../src/tower-eval/src/tower_eval/policies/xpolicylab.py",
        "            client._step += 1",
        "            pass",
    ),
    (
        "policy-replay-request-limit",
        "replay.py",
        "concurrent_request_limit=1 if policy else min(16, env.n)",
        "concurrent_request_limit=1 if policy else 0",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", help="Run one component and its unmutated test baseline")
    parser.add_argument("--partition", help="Disjoint mutation shard INDEX/COUNT, zero based")
    args = parser.parse_args()
    selected = [row for row in MUTATIONS if not args.prefix or row[0].startswith(args.prefix)]
    if not selected:
        parser.error("No mutations selected")
    if args.partition:
        index, count = map(int, args.partition.split("/"))
        if not 0 <= index < count:
            parser.error("Expected 0 <= INDEX < COUNT")
        selected = selected[index::count]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="tower-mutation-") as scratch:

        def run(name):
            env = dict(
                os.environ,
                PYTHONPYCACHEPREFIX=str(Path(scratch) / name),
                PYTHONDONTWRITEBYTECODE="1",
            )
            tests = [
                "packages/tower-sim/tests",
                "tests/characterization/test_candidate_replay.py",
                "tests/characterization/test_gpu_admission.py",
                "tests/characterization/test_gpu_host_profile.py",
                "tests/characterization/test_gpu_assemble.py",
            ]
            # The broad baseline covers non-scene tests; scene-baseline covers
            # the selected scene regressions. Mutants run their owning tests.
            if name.startswith("contrib-"):
                tests = [
                    "tests/test_public_assets.py",
                    "tests/test_public_data.py",
                    "tests/test_public_download_retry.py",
                    "tests/test_public_review_contracts.py",
                    "tests/test_public_workflows.py",
                ]
            elif name.startswith("purge-"):
                tests = [
                    "tests/test_purge_contract.py",
                    "tests/test_purge_redaction.py",
                    "tests/test_purge_ik_identity.py",
                    "tests/test_purge_archive.py",
                ]
            elif name.startswith("public-"):
                tests = [
                    f"packages/tower-sim/tests/test_public_{part}.py"
                    for part in (
                        "default",
                        "assets",
                        "paths",
                        "verification",
                        "status",
                        "layout",
                        "importers",
                        "runtime",
                        "worker",
                        "entrypoints",
                    )
                ]
            elif name == "replay-patch-visibility":
                tests = [
                    "packages/tower-sim/tests/test_replay_move.py::test_renderer_factory_observes_patch_after_environment_import"
                ]
            elif name.startswith("replay-closure-"):
                tests = ["packages/tower-sim/tests/test_import_closure.py"]
            elif name.startswith("replay-alias-") or name in {
                "replay-renderer-order",
                "replay-module-constant",
                "replay-cli-forwarding",
            }:
                tests = ["packages/tower-sim/tests/test_replay_move.py"]
            elif name in {
                "config-unknown-v1",
                "config-receipt-model",
                "config-receipt-generation",
                "config-default-value",
                "config-default-type",
                "config-v1-refusal",
                "config-generation-edge",
                "config-tasks-alias",
                "config-tasks-patch",
                "config-retained-tasks-guard",
            }:
                tests = ["packages/tower-sim/tests/test_config_split.py"]
            elif name.startswith("startup-"):
                tests = ["packages/tower-sim/tests/test_renderer_startup.py"]
            elif name.startswith("scene-"):
                tests = [
                    "packages/tower-sim/tests/test_scene_v2.py::" + test
                    for test in (
                        sorted({SCENE_TESTS[row[0]] for row in selected if row[0] in SCENE_TESTS})
                        if name == "scene-baseline"
                        else [SCENE_TESTS[name]]
                    )
                ]
            elif name.startswith("backend-") or name == "reference-tag-refusal":
                tests = ["packages/tower-sim/tests/test_backend_move.py"]
            elif name == "policy-baseline":
                tests = [
                    f"packages/tower-sim/tests/test_{part}.py"
                    for part in (
                        "policy_evaluation",
                        "policy_lifecycle",
                        "policy_protocol",
                        "policy_review_fixes",
                        "control_equivalence",
                        "pure_equivalence",
                    )
                ]
            elif name != "baseline":
                filename = next(row[1] for row in MUTATIONS if row[0] == name)
                if filename in {"bootstrap.py", "environment.py", "app.py", "scene.py"}:
                    owners = ("construction", "construction_ast", "resources_drives")
                elif filename in {"replay.py", "resources.py", "contacts.py", "source_receipt.py"}:
                    owners = (
                        "replay_lifecycle",
                        "policy_lifecycle",
                        "policy_review_fixes",
                        "physical_evidence",
                    )
                elif name.startswith("policy-"):
                    owners = (
                        "policy_evaluation",
                        "policy_lifecycle",
                        "policy_protocol",
                        "policy_review_fixes",
                        "control_equivalence",
                        "pure_equivalence",
                    )
                else:
                    owners = ()
                if owners:
                    tests = [f"packages/tower-sim/tests/test_{part}.py" for part in owners]
                # Maintainer entry/receipt checks are deliberately independent of
                # the runtime module test selection above.
                if (
                    not owners
                    or filename == "source_receipt.py"
                    or filename.endswith("/candidate_policy.py")
                ):
                    tests = [
                        "tests/characterization/test_candidate_replay.py",
                        "tests/characterization/test_gpu_admission.py",
                        "tests/characterization/test_gpu_host_profile.py",
                        "tests/characterization/test_gpu_assemble.py",
                    ] + (tests if owners else [])
            if name.startswith("policy-") or name == "baseline":
                tests.append("src/tower-eval/tests/unit/test_ws_transport_parity.py")
            if name == "policy-atomic-infer":
                tests = [
                    "src/tower-eval/tests/unit/test_ws_transport_parity.py::test_atomic_infer_prevents_interleaved_update"
                ]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    *tests,
                    "-q",
                    "--tb=short",
                    *(
                        ["--archive-root", env["TOWER_ARCHIVE_ROOT"]]
                        if env.get("TOWER_ARCHIVE_ROOT")
                        else []
                    ),
                    *(
                        ["-x"]
                        if name not in {"baseline", "scene-baseline", "purge-baseline"}
                        else []
                    ),
                    "--ignore=packages/tower-sim/tests/test_bench_scoring.py",
                    *(
                        ["--ignore=packages/tower-sim/tests/test_scene_v2.py"]
                        if not name.startswith("scene-")
                        else []
                    ),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            (args.output / f"{name}.log").write_text(result.stdout + result.stderr)
            return result.returncode

        if args.prefix == "policy-":
            if run("policy-baseline") != 0:
                raise RuntimeError("Unmutated policy suite must pass first")
        else:
            for baseline in ("contrib-baseline", "purge-baseline", "baseline"):
                if run(baseline) != 0:
                    raise RuntimeError(f"Unmutated {baseline} must pass first")
        if any(row[0] in SCENE_TESTS for row in selected) and run("scene-baseline") != 0:
            raise RuntimeError("Unmutated selected scene regressions must pass first")
        for name, filename, old, new in selected:
            path = (BENCH / filename).resolve()
            original = path.read_bytes()
            source = original.decode()
            if source.count(old) != 1:
                raise RuntimeError(f"Expected exactly one mutation site: {name}")
            try:
                path.write_text(source.replace(old, new))
                code = run(name)
                results.append(
                    dict(
                        name=name,
                        file=str(path.relative_to(ROOT)),
                        line=source[: source.index(old)].count("\n") + 1,
                        sites=source.count(old),
                        pytest_exit=code,
                        caught=code == 1,  # collection/crash is not detection
                    )
                )
                print(f"{'CAUGHT' if code == 1 else 'MISSED'} {name} (exit {code})", flush=True)
            finally:
                path.write_bytes(original)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    missed = [row["name"] for row in results if not row["caught"]]
    if missed:
        raise RuntimeError(f"Mutations not caught by tests: {', '.join(missed)}")
    print(f"{len(results)}/{len(selected)} mutations caught; source bytes restored", flush=True)


if __name__ == "__main__":
    main()
