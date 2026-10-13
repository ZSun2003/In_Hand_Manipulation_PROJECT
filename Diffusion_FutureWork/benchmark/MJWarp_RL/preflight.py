"""Fail-closed AutoDL preflight for Diffusion, IK, and native MJWarp."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from typing import Any
from xml.etree import ElementTree

from . import contract
from .contract import ACTION_DIM
from .curriculum import PHASE
from .hpc_config import DEFAULT_CONFIG
from .mjwarp_backend import PINNED_RUNTIME
from .platform_adapter import TASK_ROOT_HEIGHT_M, model_xmls_for_task

MINIMUM_DRIVER = "575.51.03"
MINIMUM_GLIBC = "2.28"


def _numeric_version(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for token in value.strip().split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        result.append(int(digits))
    if not result:
        raise ValueError(f"invalid numeric version: {value!r}")
    return tuple(result)


def _version_at_least(actual: str, required: str) -> bool:
    left, right = _numeric_version(actual), _numeric_version(required)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (
        width - len(right)
    )


def _driver_version() -> str:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "nvidia-smi failed"
        )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no driver version")
    return values[0]


def configure_cublas_workspace(*, deterministic: bool) -> str | None:
    """Establish the CuBLAS contract before any CUDA context is created."""

    if not deterministic:
        return None
    workspace = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
    )
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG to be "
            "':4096:8' or ':16:8'"
        )
    return workspace


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_runtime(
    *,
    strict: bool = False,
    task: str = DEFAULT_CONFIG.task,
    device: str | None = None,
    deterministic_torch: bool | None = None,
) -> dict[str, Any]:
    """Inspect without stepping physics; strict mode targets formal HPC."""

    selected_task = contract.validate_task(task)
    expected_root_height_m = float(TASK_ROOT_HEIGHT_M[selected_task])
    selected_device = device or DEFAULT_CONFIG.device
    selected_determinism = (
        DEFAULT_CONFIG.deterministic_torch
        if deterministic_torch is None
        else bool(deterministic_torch)
    )
    errors: list[str] = []
    model_shards: list[dict[str, Any]] = []
    model_xmls = model_xmls_for_task(selected_task)
    expected_paths = tuple(model_xmls.get(name) for name in contract.OBJECTS)
    if any(path is None for path in expected_paths):
        errors.append("task model bundle must contain every object in contract order")
    else:
        try:
            import mujoco

            object_joint_names = {
                name: f"joint:object_{name}" for name in contract.OBJECTS
            }
            for object_index, (object_name, model_path) in enumerate(
                zip(contract.OBJECTS, expected_paths)
            ):
                shard_errors: list[str] = []
                info: dict[str, Any] = {
                    "object_index": object_index,
                    "object": object_name,
                    "task": selected_task,
                    "expected_root_height_m": expected_root_height_m,
                    "path": str(model_path),
                }
                try:
                    option = (
                        ElementTree.parse(model_path)
                        .getroot()
                        .find("option")
                    )
                    solver = option.get("solver") if option is not None else None
                    jacobian = (
                        option.get("jacobian") if option is not None else None
                    )
                    tolerance_text = (
                        option.get("tolerance")
                        if option is not None
                        else None
                    )
                    tolerance = (
                        float(tolerance_text)
                        if tolerance_text is not None
                        else None
                    )
                    info.update(
                        {
                            "solver": solver,
                            "jacobian": jacobian,
                            "tolerance": tolerance,
                        }
                    )
                    if solver != contract.FORMAL_SOLVER:
                        shard_errors.append("XML solver is not Newton")
                    if jacobian != contract.FORMAL_JACOBIAN:
                        shard_errors.append("XML jacobian is not dense")
                    if tolerance != contract.FORMAL_XML_SOLVER_TOLERANCE:
                        shard_errors.append(
                            "XML tolerance does not request the formal value"
                        )

                    model = mujoco.MjModel.from_xml_path(str(model_path))
                    runtime_solver = int(model.opt.solver)
                    runtime_jacobian = int(model.opt.jacobian)
                    resolved_sparse = bool(mujoco.mj_isSparse(model))
                    model_nv = int(model.nv)
                    root_id = int(
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_BODY, "root"
                        )
                    )
                    compiled_root_height_m = (
                        float(model.body_pos[root_id, 2])
                        if root_id >= 0
                        else None
                    )
                    present_object_joints = [
                        name
                        for name, joint_name in object_joint_names.items()
                        if int(
                            mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_JOINT,
                                joint_name,
                            )
                        )
                        >= 0
                    ]
                    plate_present = int(
                        mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            "joint:object_plate",
                        )
                    ) >= 0
                    info.update(
                        {
                            "nq": int(model.nq),
                            "nv": model_nv,
                            "nu": int(model.nu),
                            "dense_newton_max_nv": (
                                contract.FORMAL_DENSE_NEWTON_MAX_NV
                            ),
                            "runtime_solver_option": runtime_solver,
                            "runtime_jacobian_option": runtime_jacobian,
                            "resolved_sparse_jacobian": resolved_sparse,
                            "present_training_object_joints": (
                                present_object_joints
                            ),
                            "plate_free_joint_present": plate_present,
                            "compiled_root_height_m": compiled_root_height_m,
                        }
                    )
                    if not 0 < model_nv <= contract.FORMAL_DENSE_NEWTON_MAX_NV:
                        shard_errors.append("compiled model violates nv <= 60")
                    if runtime_solver != int(
                        mujoco.mjtSolver.mjSOL_NEWTON
                    ):
                        shard_errors.append("compiled solver is not Newton")
                    if runtime_jacobian != int(
                        mujoco.mjtJacobian.mjJAC_DENSE
                    ) or resolved_sparse:
                        shard_errors.append(
                            "compiled model does not resolve dense Jacobian"
                        )
                    if present_object_joints != [object_name]:
                        shard_errors.append(
                            "compiled shard does not contain exactly its "
                            "expected free object"
                        )
                    if plate_present:
                        shard_errors.append(
                            "compiled shard contains the untrained plate"
                        )
                    if (
                        root_id < 0
                        or int(model.body_jntnum[root_id]) != 0
                        or compiled_root_height_m is None
                        or abs(compiled_root_height_m - expected_root_height_m)
                        > 1.0e-12
                    ):
                        shard_errors.append(
                            "compiled root height does not match the task bundle"
                        )
                except Exception as exc:
                    shard_errors.append(f"inspection failed: {exc}")
                info["ok"] = not shard_errors
                info["errors"] = shard_errors
                model_shards.append(info)
                errors.extend(
                    f"{object_name} RL shard: {message}"
                    for message in shard_errors
                )
        except Exception as exc:
            errors.append(f"four-shard model inspection failed: {exc}")
    model_xml_info = (
        dict(model_shards[0])
        if model_shards
        else {"path": str(expected_paths[0]), "ok": False}
    )
    try:
        configure_cublas_workspace(deterministic=selected_determinism)
    except RuntimeError as exc:
        errors.append(str(exc))
    packages = {name: _version(name) for name in PINNED_RUNTIME}
    for name, expected in PINNED_RUNTIME.items():
        actual = packages[name]
        if actual is None:
            errors.append(f"missing package: {name}=={expected}")
        elif actual.split("+", 1)[0] != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    if strict and platform.system() != "Linux":
        errors.append("formal native MJWarp training requires Linux")
    if strict and not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        errors.append("formal runtime requires Python 3.10, 3.11, or 3.12")
    libc_name, libc_version = platform.libc_ver()
    if strict and (
        libc_name.lower() != "glibc"
        or not _version_at_least(libc_version, MINIMUM_GLIBC)
    ):
        errors.append(
            f"formal runtime requires glibc >= {MINIMUM_GLIBC}, "
            f"found {libc_name} {libc_version}"
        )

    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "bf16_supported": (
                torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            ),
        }
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        else:
            torch_device = torch.device(selected_device)
            properties = torch.cuda.get_device_properties(torch_device)
            capability = torch.cuda.get_device_capability(torch_device)
            torch_info.update(
                {
                    "device": str(torch_device),
                    "name": properties.name,
                    "memory_bytes": int(properties.total_memory),
                    "compute_capability": list(capability),
                    "arch_list": torch.cuda.get_arch_list(),
                }
            )
            normalized_name = "".join(
                character
                for character in properties.name.upper()
                if character.isalnum()
            )
            if strict and not (
                "RTXPRO6000" in normalized_name
                and "BLACKWELL" in normalized_name
                and "MAXQ" not in normalized_name
            ):
                errors.append(
                    "formal profile expects an RTX PRO 6000 Blackwell GPU"
                )
            if strict and properties.total_memory < 90 * 1024**3:
                errors.append("formal profile expects at least 90 GiB VRAM")
            if strict and capability != (12, 0):
                errors.append("formal profile expects Blackwell sm_120")
            if strict and "sm_120" not in torch.cuda.get_arch_list():
                errors.append("PyTorch wheel does not contain sm_120 kernels")
            if strict and str(torch.version.cuda) != "12.8":
                errors.append(
                    f"PyTorch wheel must use CUDA 12.8, found {torch.version.cuda}"
                )
            if not torch.cuda.is_bf16_supported():
                errors.append("BF16 is not supported by the selected GPU")
            if strict:
                try:
                    driver = _driver_version()
                    torch_info["driver_version"] = driver
                    if not _version_at_least(driver, MINIMUM_DRIVER):
                        errors.append(
                            f"NVIDIA driver must be >= {MINIMUM_DRIVER}, "
                            f"found {driver}"
                        )
                except Exception as exc:
                    errors.append(f"NVIDIA driver inspection failed: {exc}")
    except Exception as exc:  # pragma: no cover - depends on HPC runtime.
        errors.append(f"PyTorch CUDA inspection failed: {exc}")

    warp_info: dict[str, Any] = {}
    try:
        import warp as wp

        wp.init()
        warp_device = wp.get_device(selected_device)
        warp_info = {
            "device": str(warp_device),
            "is_cuda": bool(getattr(warp_device, "is_cuda", False)),
            "architecture": getattr(warp_device, "arch", None),
            "toolkit": list(wp.get_cuda_toolkit_version()),
            "supported_architectures": list(wp.get_cuda_supported_archs()),
        }
        if not warp_info["is_cuda"]:
            errors.append("Warp cannot see the selected CUDA device")
        if strict and int(warp_info["architecture"] or 0) != 120:
            errors.append("Warp device must expose sm_120")
        if strict and tuple(warp_info["toolkit"]) != (12, 9):
            errors.append("Warp 1.14 must use its CUDA 12.9 toolkit")
        if strict and 120 not in warp_info["supported_architectures"]:
            errors.append("Warp NVRTC cannot compile sm_120")
    except Exception as exc:  # pragma: no cover - depends on HPC runtime.
        errors.append(f"Warp inspection failed: {exc}")

    return {
        "ok": not errors,
        "strict": bool(strict),
        "task": selected_task,
        "root_height_m": expected_root_height_m,
        "platform": platform.platform(),
        "python": sys.version,
        "glibc": {
            "name": libc_name,
            "version": libc_version,
        },
        "selected_device": selected_device,
        "deterministic_torch": selected_determinism,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "packages": packages,
        "physics_layout": contract.PHYSICS_LAYOUT,
        "model_xml": model_xml_info,
        "model_shards": model_shards,
        "torch": torch_info,
        "warp": warp_info,
        "errors": errors,
    }


def run_smoke(
    *,
    num_envs: int = 8,
    phase: str = PHASE,
    task: str = DEFAULT_CONFIG.task,
    capture_graph: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Exercise native physics, Diffusion grads/DDIM, runtime, and plane bounds."""

    if phase != PHASE:
        raise ValueError(f"phase must be {PHASE!r}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    selected_task = contract.validate_task(task)
    selected_device = device or DEFAULT_CONFIG.device
    configure_cublas_workspace(
        deterministic=DEFAULT_CONFIG.deterministic_torch
    )
    import torch
    torch.cuda.set_device(torch.device(selected_device))
    torch.manual_seed(DEFAULT_CONFIG.seed)
    torch.cuda.manual_seed_all(DEFAULT_CONFIG.seed)
    torch.use_deterministic_algorithms(
        DEFAULT_CONFIG.deterministic_torch
    )

    from .diffusion_policy import DiffusionPolicy
    from .diffusion_runtime import (
        BatchedDiffusionController,
        BatchedObservationHistory,
    )
    from .env import OBSERVATION_DIM, action_mask_from_observation, make_env
    from .platform_adapter import build_platform_spec
    from .sharded_env import formal_sharded_health_is_valid

    config = replace(
        DEFAULT_CONFIG,
        task=selected_task,
        num_envs=num_envs,
        num_eval_envs=num_envs,
        device=selected_device,
        cuda_graph=capture_graph,
        # Smoke sizes do not change the formal training profile.
        total_steps=max(
            DEFAULT_CONFIG.episode_length * num_envs,
            num_envs,
        ),
    )
    # The official-budget validator intentionally rejects smoke budgets, so
    # construct physics directly without calling config.validate().
    spec = build_platform_spec(selected_task)
    environment = make_env(
        phase,
        config=config,
        platform_spec=spec,
        worlds=num_envs,
        device=selected_device,
        require_linux=True,
        validate_official_config=False,
    )
    try:
        observation = environment.reset()
        if capture_graph:
            environment.backend.prepare_cuda_graph()
            if not environment.backend._graph_ready:
                raise RuntimeError("CUDA Graph was requested but not captured")
            observation = environment.reset()
        before = environment.backend.tensor("time").clone()
        zero_action = torch.zeros(
            (num_envs, ACTION_DIM),
            device=environment.device,
            dtype=torch.float32,
        )
        shards = tuple(getattr(environment, "shards", (environment,)))
        if len(shards) != contract.OBJECT_SHARD_COUNT:
            raise RuntimeError("smoke environment did not create four shards")
        for object_index, shard in enumerate(shards):
            backend = shard.backend
            torch_stream_handle = int(
                backend.torch_stream.cuda_stream
            )
            warp_stream_handle = int(backend.wp_stream.cuda_stream)
            if torch_stream_handle != warp_stream_handle:
                raise RuntimeError(
                    f"{contract.OBJECTS[object_index]} Torch/Warp streams "
                    "do not share one native handle"
                )
            qpos_view = backend.tensor("qpos")
            if qpos_view.data_ptr() != int(backend.data.qpos.ptr):
                raise RuntimeError(
                    f"{contract.OBJECTS[object_index]} qpos is not zero-copy"
                )
            if (
                backend._reset_mask.data_ptr()
                != int(backend._reset_mask_wp.ptr)
            ):
                raise RuntimeError(
                    f"{contract.OBJECTS[object_index]} reset mask is not "
                    "zero-copy"
                )

        graph_replays = 4
        time_samples = []
        transition = None
        for _ in range(graph_replays):
            transition = environment.step(zero_action)
            time_samples.append(
                environment.backend.tensor("time").clone()
            )
        if transition is None:
            raise RuntimeError("native smoke executed no control step")
        expected_dt = environment.backend.control_dt
        if observation.shape != (num_envs, OBSERVATION_DIM):
            raise RuntimeError("native observation has the wrong shape")
        if transition.observation.shape != (num_envs, OBSERVATION_DIM):
            raise RuntimeError("native step observation has the wrong shape")
        time_trace = torch.stack((before, *time_samples), dim=0)
        if not torch.allclose(
            time_trace[1:] - time_trace[:-1],
            torch.full_like(time_trace[1:], expected_dt),
            atol=2.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "repeated 40-substep graph replay did not advance by 20 ms"
            )
        if not torch.isfinite(transition.observation).all():
            raise FloatingPointError("native observation is non-finite")

        settings = DEFAULT_CONFIG.diffusion
        policy = DiffusionPolicy(
            obs_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            obs_horizon=settings.obs_horizon,
            horizon=settings.action_horizon,
            num_diffusion_steps=settings.diffusion_steps,
            hidden_dim=settings.hidden_dim,
            depth=settings.depth,
            dropout=settings.dropout,
        ).to(environment.device)
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        training_observation = transition.observation
        observation_history = training_observation[:, None, :].expand(
            -1, settings.obs_horizon, -1
        ).contiguous()
        target_action = (
            2.0
            * torch.rand(
                (
                    num_envs,
                    settings.action_horizon,
                    ACTION_DIM,
                ),
                device=environment.device,
            )
            - 1.0
        )
        optimizer.zero_grad(set_to_none=True)
        use_bfloat16 = (
            settings.use_bfloat16
            and torch.cuda.is_bf16_supported()
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            diffusion_loss = policy.compute_loss(
                observation_history,
                target_action,
            )
        if not bool(torch.isfinite(diffusion_loss)):
            raise FloatingPointError("Diffusion loss is non-finite")
        diffusion_loss.backward()
        diffusion_gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), settings.max_grad_norm
        )
        if not bool(torch.isfinite(diffusion_gradient_norm)):
            raise FloatingPointError("Diffusion gradient is non-finite")
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all())
            for parameter in policy.parameters()
        ):
            raise FloatingPointError(
                "Diffusion optimizer produced non-finite parameters"
            )

        history = BatchedObservationHistory(
            worlds=num_envs,
            obs_horizon=settings.obs_horizon,
            obs_dim=OBSERVATION_DIM,
            device=environment.device,
        )
        all_worlds = torch.ones(
            num_envs, device=environment.device, dtype=torch.bool
        )
        history.reset(all_worlds, training_observation)
        controller = BatchedDiffusionController(
            policy,
            worlds=num_envs,
            inference_steps=settings.inference_steps,
            replan_interval=settings.replan_interval,
            seed=DEFAULT_CONFIG.seed,
            action_clip=True,
        )
        controller.reset(all_worlds, history.values)
        if controller.cursor != 0 or controller.sample_count != 1:
            raise RuntimeError("full Diffusion reset did not prime one chunk")
        ddim_action = controller.act(history.values)
        if tuple(ddim_action.shape) != (num_envs, ACTION_DIM):
            raise RuntimeError("DDIM action has the wrong shape")
        if not bool(torch.isfinite(ddim_action).all()):
            raise FloatingPointError("DDIM action is non-finite")
        if controller.cursor != 1:
            raise RuntimeError("Diffusion runtime cursor did not advance")

        plan_before_partial_reset = controller.plan.clone()
        if num_envs > 1:
            partial_mask = torch.zeros(
                num_envs,
                device=environment.device,
                dtype=torch.bool,
            )
            partial_mask[::2] = True
            history.reset(partial_mask, training_observation)
            controller.reset(partial_mask, history.values)
            if controller.cursor != 1:
                raise RuntimeError(
                    "partial Diffusion reset moved the shared cursor"
                )
            if not torch.equal(
                controller.plan, plan_before_partial_reset
            ):
                raise RuntimeError(
                    "partial Diffusion reset changed unrelated base plans"
                )
            if not bool(controller.override_active[partial_mask].all()):
                raise RuntimeError(
                    "partial Diffusion reset did not install overrides"
                )
            if bool(controller.override_active[~partial_mask].any()):
                raise RuntimeError(
                    "partial Diffusion reset changed unrelated worlds"
                )
            runtime_reset_mode = "partial-world override"
        else:
            history.reset(all_worlds, training_observation)
            controller.reset(all_worlds, history.values)
            if controller.cursor != 0:
                raise RuntimeError(
                    "full Diffusion reset did not rewind the cursor"
                )
            runtime_reset_mode = "full-world rewind"

        safe_action = torch.nan_to_num(
            ddim_action,
            nan=0.0,
            posinf=contract.NORMALIZED_ACTION_MAX,
            neginf=contract.NORMALIZED_ACTION_MIN,
        ).clamp(
            contract.NORMALIZED_ACTION_MIN,
            contract.NORMALIZED_ACTION_MAX,
        )
        policy_action_mask = action_mask_from_observation(
            training_observation
        ).to(safe_action.dtype)
        safe_action *= policy_action_mask
        world_object_index = environment.object_index.to(torch.long)
        learned_world = torch.zeros_like(
            world_object_index, dtype=torch.bool
        )
        for object_index, object_kind in enumerate(contract.OBJECTS):
            if contract.grasp_plane_is_learned(object_kind):
                learned_world |= world_object_index == object_index
        fixed_world = ~learned_world
        height_and_yaw = slice(
            contract.TIP_HEIGHT_SLICE.start, contract.YAW_INDEX + 1
        )
        if bool(fixed_world.any()) and not bool(
            (policy_action_mask[fixed_world, height_and_yaw] == 0.0).all()
        ):
            raise RuntimeError(
                "fixed-plane object action mask did not disable all four "
                "tip-height channels and yaw"
            )
        if bool(learned_world.any()) and not bool(
            (policy_action_mask[learned_world, height_and_yaw] == 1.0).all()
        ):
            raise RuntimeError(
                "learned-fruit action mask disabled tip height or yaw"
            )
        if bool(fixed_world.any()) and not bool(
            (safe_action[fixed_world, height_and_yaw] == 0.0).all()
        ):
            raise RuntimeError(
                "masked Diffusion command retained fixed-plane height/yaw"
            )
        diffusion_transition = environment.step(safe_action)
        history.append(diffusion_transition.observation)
        snapshot = environment.snapshot()
        plane_fraction = snapshot["grasp_plane_fraction"]
        plane_compliant = snapshot["grasp_plane_compliant"].to(
            torch.bool
        )
        actual_plane_fraction = snapshot[
            "actual_grasp_plane_fraction"
        ]
        actual_plane_compliant = snapshot[
            "actual_grasp_plane_compliant"
        ].to(torch.bool)
        actual_plane_finite = snapshot[
            "actual_grasp_plane_finite"
        ].to(torch.bool)
        actual_plane_relevant = snapshot[
            "actual_grasp_plane_relevant"
        ].to(torch.bool)
        commanded_plane_error_m = snapshot[
            "commanded_grasp_plane_error_m"
        ]
        actual_plane_error_m = snapshot["actual_grasp_plane_error_m"]
        object_height_m = snapshot["object_height_m"]
        snapshot_object_index = snapshot["object_index"].to(torch.long)
        commanded_rule_compliant = torch.zeros_like(
            plane_compliant, dtype=torch.bool
        )
        actual_rule_compliant = torch.zeros_like(
            actual_plane_compliant, dtype=torch.bool
        )
        per_object_plane: dict[str, Any] = {}
        numerical_tolerance = 2.0e-6
        for object_index, object_kind in enumerate(contract.OBJECTS):
            selected = snapshot_object_index == object_index
            if not bool(selected.any()):
                continue
            low, high = contract.grasp_plane_fraction_limits(object_kind)
            target = contract.grasp_plane_nominal_fraction(object_kind)
            learned = contract.grasp_plane_is_learned(object_kind)
            if learned:
                commanded_selected = (
                    torch.isfinite(plane_fraction[selected])
                    & (
                        plane_fraction[selected]
                        >= low - numerical_tolerance
                    )
                    & (
                        plane_fraction[selected]
                        <= high + numerical_tolerance
                    )
                )
                relevant_selected = actual_plane_relevant[selected]
                relevant_fraction_valid = (
                    torch.isfinite(actual_plane_fraction[selected])
                    & (
                        actual_plane_fraction[selected]
                        >= low - numerical_tolerance
                    )
                    & (
                        actual_plane_fraction[selected]
                        <= high + numerical_tolerance
                    )
                )
                actual_selected = (
                    actual_plane_finite[selected]
                    & actual_plane_compliant[selected]
                    & ((~relevant_selected) | relevant_fraction_valid)
                )
            else:
                commanded_selected = (
                    torch.isfinite(plane_fraction[selected])
                    & torch.isfinite(commanded_plane_error_m[selected])
                    & (
                        (plane_fraction[selected] - target).abs()
                        <= numerical_tolerance
                    )
                    & (
                        commanded_plane_error_m[selected].abs()
                        <= numerical_tolerance
                        * object_height_m[selected]
                        + 1.0e-9
                    )
                )
                relevant_selected = actual_plane_relevant[selected]
                relevant_error_valid = (
                    torch.isfinite(actual_plane_error_m[selected])
                    & (
                        actual_plane_error_m[selected].abs()
                        <= contract.FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M
                        + 1.0e-9
                    )
                )
                # During approach the physical tips are deliberately still at
                # pregrasp.  Once the endpoint-aware relevance gate activates,
                # verify the explicit metric error as well as the environment's
                # object-specific compliance boolean.
                actual_selected = (
                    actual_plane_finite[selected]
                    & actual_plane_compliant[selected]
                    & ((~relevant_selected) | relevant_error_valid)
                )
            commanded_rule_compliant[selected] = commanded_selected
            actual_rule_compliant[selected] = actual_selected
            relevant_actual_selected = actual_selected[relevant_selected]
            per_object_plane[object_kind] = {
                "mode": "learned" if learned else "fixed",
                "fraction_bounds": [float(low), float(high)],
                "target_fraction": float(target),
                "commanded_minimum_fraction": float(
                    plane_fraction[selected].min().detach().cpu()
                ),
                "commanded_maximum_fraction": float(
                    plane_fraction[selected].max().detach().cpu()
                ),
                "physical_compliance_rate": (
                    float(
                        relevant_actual_selected.to(torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                    )
                    if int(relevant_actual_selected.numel())
                    else None
                ),
                "physical_relevant_worlds": int(
                    actual_plane_relevant[selected].sum().detach().cpu()
                ),
                "maximum_relevant_physical_error_m": (
                    float(
                        actual_plane_error_m[
                            selected & actual_plane_relevant
                        ]
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    if bool((selected & actual_plane_relevant).any())
                    else None
                ),
            }
        if not bool(
            (plane_compliant & commanded_rule_compliant).all()
        ):
            raise RuntimeError(
                "commanded grasp plane violated an object-specific fixed or "
                "learned rule"
            )
        if not bool(actual_rule_compliant.all()):
            raise RuntimeError(
                "physical hard-tip plane violated the fruit 40%-60% band or "
                "the rigid-object 2 mm fixed-plane tolerance"
            )
        relevant_actual_rule_compliant = actual_rule_compliant[
            actual_plane_relevant
        ]
        if "grasp_plane_fraction" not in diffusion_transition.info:
            raise RuntimeError(
                "step info is missing grasp-plane fraction telemetry"
            )
        if "grasp_plane_compliant" not in diffusion_transition.info:
            raise RuntimeError(
                "step info is missing grasp-plane compliance telemetry"
            )
        for key in (
            "commanded_grasp_plane_error_m",
            "actual_grasp_plane_fraction",
            "actual_grasp_plane_error_m",
            "actual_grasp_plane_relevant",
            "actual_grasp_plane_within_bounds",
            "actual_grasp_plane_compliant",
        ):
            if key not in diffusion_transition.info:
                raise RuntimeError(
                    f"step info is missing {key} telemetry"
                )
        health = environment.backend.health()
        solver_invalid = not formal_sharded_health_is_valid(
            health,
            expected_world_counts=tuple(environment.world_counts),
        )
        unhealthy = (
            not health["finite"]
            or solver_invalid
            or health["sticky_nonfinite"]
            or health["sticky_overflow"]
            or health["sticky_contacts_full"]
            or health["sticky_constraints_full"]
            or health["sticky_external_nonfinite"]
        )
        if unhealthy:
            raise FloatingPointError(
                f"native MJWarp health check failed: {health}"
            )
        return {
            "ok": True,
            "num_envs": num_envs,
            "phase": phase,
            "observation_shape": list(observation.shape),
            "control_dt_s": expected_dt,
            "cuda_graph": capture_graph,
            "cuda_graph_count": contract.CUDA_GRAPHS_PER_GLOBAL_BATCH,
            "object_world_counts": list(environment.world_counts),
            "graph_replays": graph_replays,
            "torch_warp_stream_handle_match": True,
            "zero_copy_pointer_match": True,
            "diffusion": {
                "loss": float(diffusion_loss.detach().cpu()),
                "gradient_norm": float(
                    diffusion_gradient_norm.detach().cpu()
                ),
                "ddim_action_shape": list(ddim_action.shape),
                "ddim_action_finite": True,
                "bfloat16_autocast": use_bfloat16,
            },
            "diffusion_runtime": {
                "obs_horizon": settings.obs_horizon,
                "action_horizon": settings.action_horizon,
                "inference_steps": settings.inference_steps,
                "replan_interval": settings.replan_interval,
                "cursor_advanced": True,
                "masked_reset": runtime_reset_mode,
            },
            "grasp_plane": {
                "learned_fraction_bounds": [
                    contract.GRASP_PLANE_MIN_FRACTION,
                    contract.GRASP_PLANE_MAX_FRACTION,
                ],
                "learned_objects": [
                    object_kind
                    for object_kind in contract.OBJECTS
                    if object_kind in contract.LEARNED_GRASP_PLANE_OBJECTS
                ],
                "fixed_fractions": {
                    object_kind: float(
                        contract.FIXED_GRASP_PLANE_FRACTIONS[object_kind]
                    )
                    for object_kind in contract.OBJECTS
                    if object_kind in contract.FIXED_GRASP_PLANE_FRACTIONS
                },
                "fixed_yaw_rad": contract.FIXED_GRASP_PLANE_YAW_RAD,
                "fixed_physical_tolerance_m": (
                    contract.FIXED_GRASP_PLANE_PHYSICAL_TOLERANCE_M
                ),
                "by_object": per_object_plane,
                "minimum_fraction": float(
                    plane_fraction.min().detach().cpu()
                ),
                "maximum_fraction": float(
                    plane_fraction.max().detach().cpu()
                ),
                "compliance_rate": float(
                    plane_compliant.to(torch.float32)
                    .mean()
                    .detach()
                    .cpu()
                ),
                "telemetry_present": True,
                "actual_measurement": "physical_hard_tip_positions",
                "actual_fraction_finite": True,
                "actual_validation": "gating_smoke_only",
                "actual_relevant_sample_count": int(
                    relevant_actual_rule_compliant.numel()
                ),
                "actual_compliance_rate": (
                    float(
                        relevant_actual_rule_compliant.to(torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                    )
                    if int(relevant_actual_rule_compliant.numel())
                    else None
                ),
            },
            "health": health,
        }
    finally:
        environment.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--task", choices=contract.TASKS, required=True)
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_runtime(
        strict=args.strict,
        task=args.task,
        device=args.device,
        deterministic_torch=DEFAULT_CONFIG.deterministic_torch,
    )
    if args.smoke and report["ok"]:
        try:
            report["smoke"] = run_smoke(
                num_envs=args.num_envs,
                phase=PHASE,
                task=args.task,
                capture_graph=not args.no_cuda_graph,
                device=args.device,
            )
        except Exception as exc:
            report["ok"] = False
            report["errors"].append(f"native smoke failed: {exc}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
