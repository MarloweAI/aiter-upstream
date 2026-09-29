# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_HIDDEN = 6144
_OUT_DIM = 2624  # q_lora 2048 | kv_lora 512 | rotary key 64
_Q_LORA_RANK = 2048
_KV_LORA_RANK = 512
_SPLITK_PLANES = 6
_SUPPORTED_M = (128, 256)
# 41 column tiles of 64 times 6 K ranges of 1024: one wave on 256 CUs.
_GRID = (246,)
_NUM_CUS = 256


@functools.cache
def _device_supported(device_index: int) -> tuple[bool, str]:
    try:
        props = torch.cuda.get_device_properties(device_index)
    except Exception as exc:  # noqa: BLE001
        return False, f"device query failed ({exc})"
    arch = getattr(props, "gcnArchName", "").split(":")[0]
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch or 'unknown'}"
    if props.multi_processor_count != _NUM_CUS:
        return False, f"{_NUM_CUS} CUs only, got {props.multi_processor_count}"
    try:
        import triton.experimental.gluon  # noqa: F401
    except ImportError as exc:
        return False, f"triton.experimental.gluon unavailable ({exc})"
    return True, ""


def fused_gemm_a16w16_qk_rmsnorm_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> tuple[bool, str]:
    """Report whether fused_gemm_a16w16_qk_rmsnorm covers these operands.

    Returns ``(True, "")`` or ``(False, reason)`` and never raises, so a model
    can call it on every forward and take its unfused path on ``False``. The op
    covers M = 128 and M = 256 only, on a 256-CU gfx950.
    """
    if x.dim() != 2 or x.shape[0] not in _SUPPORTED_M:
        return False, f"x must be [128 or 256, 6144], got {list(x.shape)}"
    expected = (
        ("x", x, (x.shape[0], _HIDDEN)),
        ("weight", weight, (_OUT_DIM, _HIDDEN)),
        ("q_weight", q_weight, (_Q_LORA_RANK,)),
        ("k_weight", k_weight, (_KV_LORA_RANK,)),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            return False, f"{name} must be bf16 {list(shape)}"
        if not tensor.is_contiguous() or tensor.data_ptr() % 16:
            return False, f"{name} must be contiguous and 16-byte aligned"
        if tensor.device != x.device:
            return False, "all tensors must be on one device"
    if not x.is_cuda:
        return False, "x must be a GPU tensor"
    return _device_supported(x.device.index)


def fused_gemm_a16w16_qk_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k_weight: torch.Tensor,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLA input projection with the q and kv RMSNorms that follow it.

    Computes ``out = x @ weight.T`` for ``x`` ``[M, 6144]`` and ``weight``
    ``[2624, 6144]``, whose output columns are q_lora (2048), kv_lora (512) and
    the rotary key (64), then ``q_out = RMSNorm(out[:, :2048]) * q_weight`` and
    ``k_out = RMSNorm(out[:, 2048:2560]) * k_weight``. All tensors are bf16.

    A Gluon kernel on 246 workgroups writes six fp32 partial planes, one per K
    range of 1024, reading every weight element once; ``splitk_reduce_qk_rmsnorm``
    adds them in a fixed order, rounds to bf16 once and applies both norms. The
    result is deterministic. The planes are a per-call fp32 workspace of
    ``6 * M * 2624`` elements (7.7 MiB at M = 128, 15.4 MiB at M = 256).

    Args:
        x: ``[M, 6144]`` activations, M = 128 or 256.
        weight: ``[2624, 6144]`` projection weight, row-major.
        q_weight: ``[2048]`` RMSNorm weight of the q columns.
        q_eps: its epsilon.
        k_weight: ``[512]`` RMSNorm weight of the kv columns.
        k_eps: its epsilon.

    Returns:
        ``(out, q_out, k_out)``: ``[M, 2624]``, ``[M, 2048]`` and ``[M, 512]``.

    Raises:
        ValueError: when ``fused_gemm_a16w16_qk_rmsnorm_supported`` rejects the
            operands. A model path checks that first and otherwise runs
            ``gemm_a16w16`` and ``fused_qk_rmsnorm``.
    """
    supported, reason = fused_gemm_a16w16_qk_rmsnorm_supported(
        x, weight, q_weight, k_weight
    )
    if not supported:
        raise ValueError(f"fused_gemm_a16w16_qk_rmsnorm: {reason}")
    _LOGGER.info(
        "FUSED_GEMM_A16W16_QK_RMSNORM: x=%s w=%s", tuple(x.shape), tuple(weight.shape)
    )

    from aiter.ops.splitk_reduce_qk_rmsnorm import splitk_reduce_qk_rmsnorm
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.fused.fused_gemm_a16w16_qk_rmsnorm import (
        _gemm_a16w16_splitk_planes_kernel,
    )

    m = x.shape[0]
    partial = torch.empty(
        (_SPLITK_PLANES, m, _OUT_DIM), device=x.device, dtype=torch.float32
    )
    _gemm_a16w16_splitk_planes_kernel[_GRID](
        x,
        weight,
        partial,
        BLOCK_M=m,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
    )
    return splitk_reduce_qk_rmsnorm(partial, q_weight, q_eps, k_weight, k_eps)
