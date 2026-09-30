"""Independent expert reference with dispatch-selected MXFP4 contracts.

Current generated FlyDSL uses 0x7fffff exponent roundup, distinct from the
historical 0x400000 fused quantizer. Never choose a reference by observed error.
"""


def native_contract(metadata):
    stage = metadata.stage1
    function = getattr(stage, "func", stage)
    name = getattr(function, "__name__", "")
    kwargs = getattr(stage, "keywords", {}) or {}
    kernel = kwargs.get("kernelName1", "")
    if name == "_mxfp4_a4w4_stage1_fw" and "_a4w4_" in kernel and "_f16in" in kernel:
        if metadata.fuse_quant != "fp4":
            raise ValueError("Generated FP4 reference requires an FP4 intermediate")
        return {
            "input_rounding": "flydsl_roundup_0x7fffff",
            "middle_rounding": "flydsl_roundup_0x7fffff",
            "fp32_middle": True,
            "output": "bf16_expert_operands",
        }
    if name == "_flydsl_stage1_wrapper" and metadata.fuse_quant == "fp4":
        return {
            "input_rounding": "hip_round_up",
            "middle_rounding": "flydsl_fused_0x400000",
            "fp32_middle": True,
            "output": "bf16_expert_operands",
        }
    # Unknown/native materialized paths need a source-verified mapping before
    # admission; absence of fuse_quant alone does not establish their numerics.
    raise ValueError(f"Unqualified native reference family: {name}/{kernel}")


def candidate_contract(variant):
    return {
        "input_rounding": "hip_round_up",
        "middle_rounding": "flydsl_fused_0x400000"
        if variant.fp32_middle
        else "hip_round_up",
        "fp32_middle": variant.fp32_middle,
        "output": "bf16_expert_operands",
    }


def quantize(x, rounding):
    from .reference import quantize_mxfp4, round_e2m1

    if rounding != "flydsl_roundup_0x7fffff":
        return quantize_mxfp4(x, scale_rounding=rounding)[0]
    import torch

    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    bits = (amax * (1.0 / 6.0)).contiguous().view(torch.int32)
    exponent = ((bits + 0x7FFFFF) >> 23).clamp(0, 254)
    scale = torch.exp2(exponent.float() - 127.0)
    return (round_e2m1(blocks / scale) * scale).reshape_as(x)


def unpack_weights(weights):
    """Invert packed weight/scale permutations independently of native kernels."""
    import torch

    from .reference import dequant_weight

    decoded = {}
    for key in ("w1", "w2"):
        tensor = weights[key]
        experts, n, width = tensor.shape
        packed = tensor.view(torch.uint8).view(experts, n // 16, width // 32, 2, 16, 16)
        packed = packed.permute(0, 1, 4, 2, 3, 5).contiguous().view(tensor.shape)
        scales = weights[key + "_scale"]
        m, sn = experts * n, scales.shape[-1]
        scales = scales.view(torch.uint8).view(m // 32, sn // 8, 4, 16, 2, 2)
        scales = scales.permute(0, 5, 3, 1, 4, 2).contiguous().view(experts, n, sn)
        decoded[key] = dequant_weight(packed, scales, format="mxfp4")
    return decoded


def expert_reference(decoded, x, ids, weights, contract):
    import torch

    from .reference import bf16_atomic_operand_reference

    qx = quantize(x, contract["input_rounding"])
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(257):
        rows, choices = torch.where(ids == expert)
        if not len(rows):
            continue
        gate, up = (qx[rows] @ decoded["w1"][expert].T).chunk(2, -1)
        middle = gate * torch.sigmoid(gate) * up
        if not contract["fp32_middle"]:
            middle = middle.bfloat16()
        middle = quantize(middle, contract["middle_rounding"])
        contribution = (middle @ decoded["w2"][expert].T) * weights[rows, choices, None]
        out[rows] += bf16_atomic_operand_reference(contribution)
    return out.bfloat16()
