# Numerical routines retained from the frozen GLM study and library reference.
# Quantizer/numerics source SHA256:
# ab6213474f1683ce7b8e7e6ea939897e5c49fde4cd0571390db9b9ba5da29276
# 5576c8f6c90838b0f9aabd2b570c6004e621f478d92d554794ac1d19010f4538
"""Independent FP32 GEMMs and quantization, with explicit BF16 atomic rounding.
No vendor quantizer, GEMM or MoE calls.

Quantization choices are explicit arguments, never inferred from batch size.
An adapter must verify these choices against captured dispatch before using them.
"""
import torch


def decode_e2m1(packed):
    raw = packed.contiguous().view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=raw.device)
    values = lut[codes & 7]
    return torch.where((codes & 8) != 0, -values, values)


def round_e2m1(x):
    """Round-to-nearest, ties-to-even, preserving signed zero and saturation."""
    a = x.float().abs().contiguous()
    edges = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device)
    code = torch.bucketize(a, edges)
    ties = (a == edges[code.clamp_max(6)]) & (code < 7)
    code = code + (ties & ((code & 1) != 0)).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    return torch.copysign(lut[code], x.float())


def decode_e8m0(scales):
    raw = scales.contiguous().view(torch.uint8)
    value = torch.exp2(raw.float()-127)
    return torch.where(raw == 255, torch.full_like(value, float('nan')), value)


def dequant_weight(packed, scales, *, format, global_scale=1.):
    values = decode_e2m1(packed)
    if format == 'mxfp4':
        block, scales32 = 32, decode_e8m0(scales)
    elif format == 'nvfp4':
        block, scales32 = 16, scales.float()
    else:
        raise ValueError('Unsupported FP4 format')
    if values.shape[-1] // block != scales32.shape[-1]:
        raise ValueError('Noncanonical scale layout or incorrect block size')
    return (values.reshape(*values.shape[:-1], -1, block) * scales32[..., None]).reshape_as(values) * global_scale


def quantize_mxfp4(x, *, scale_rounding, storage_dtype=None):
    y = x.to(storage_dtype).float() if storage_dtype is not None else x.float()
    blocks = y.reshape(*y.shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    if scale_rounding == 'hip_round_up':
        # Match the documented FP32 multiply by 1/6 before exponent rounding.
        scale_bits = (amax.clamp_min(1e-10) * (1./6.)).contiguous().view(torch.int32)
        exponent = ((scale_bits >> 23) & 255) + ((scale_bits & 0x7fffff) != 0).int()
    elif scale_rounding == 'flydsl_fused_0x400000':
        bits = amax.contiguous().view(torch.int32)
        exponent = ((((bits + 0x400000) & 0xff800000) >> 23) - 2).clamp_min(0)
    else:
        raise ValueError('Unverified MXFP4 scale rounding')
    scale = torch.exp2(exponent.float()-127.)
    result = round_e2m1(blocks / scale) * scale
    return result.reshape_as(y), exponent.squeeze(-1).to(torch.uint8)


TOLERANCES = {"rtol": 0.02, "atol": 0.02, "nrmse": 0.01}

def compare(actual, expected):
    import torch
    if actual.shape != expected.shape:
        return dict(passed=False, reason='shape mismatch')
    a, r = actual.detach().double(), expected.detach().to(actual.device).double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    if not finite:
        return dict(passed=False, finite=False, reason='NaN or infinity')
    delta = a - r
    ss = float(delta.square().sum())
    rs = float(r.square().sum())
    denom = float(a.square().sum()) + rs
    nrmse = (ss / rs) ** .5 if rs else (0. if ss == 0 else None)
    mismatch = delta.abs() > TOLERANCES['atol'] + TOLERANCES['rtol'] * r.abs()
    fraction = float(mismatch.double().mean()) if a.numel() else 0.
    return dict(passed=fraction == 0 and nrmse is not None and nrmse <= TOLERANCES['nrmse'],
                finite=True, mismatch_fraction=fraction, nrmse=nrmse,
                global_difference_D=ss / denom if denom else 0.,
                max_abs_error=float(delta.abs().max()) if a.numel() else 0.,
                tolerances=TOLERANCES)


def bf16_atomic_operand_reference(contribution):
    """Pack the weighted contribution as the BF16 atomic operand.

    The reference sums these operands in FP32, so its answer does not depend
    on an arbitrary BF16 expert-addition order. Kernel accumulation error is
    still checked against the unchanged tolerances.
    """
    return contribution.bfloat16().float()


def expert_reference(weights,x,ids,rw,fused):
    import torch
    w=weights;p={}
    # Inverse of the verified AITER 16x16 weight and E8M0 scale shuffle.
    for k in ('w1','w2'):
        t=w[k];e,n,width=t.shape
        p[k]=t.view(torch.uint8).view(e,n//16,width//32,2,16,16).permute(0,1,4,2,3,5).contiguous().view(t.shape)
        s=w[k+'_scale'];m,sn=e*n,s.shape[-1]
        p[k+'_scale']=s.view(torch.uint8).view(m//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(e,n,sn)
    qx,_=quantize_mxfp4(x.float(),scale_rounding='hip_round_up')
    out=torch.zeros_like(x,dtype=torch.float32)
    for e in range(257):
        rows,choices=torch.where(ids==e)
        if not len(rows):continue
        a=dequant_weight(p['w1'][e],p['w1_scale'][e],format='mxfp4')
        gate,up=(qx[rows]@a.T).chunk(2,-1)
        mid=gate*torch.sigmoid(gate)*up
        if not fused:mid=mid.bfloat16()
        mid,_=quantize_mxfp4(mid,scale_rounding='flydsl_fused_0x400000' if fused else 'hip_round_up')
        down=dequant_weight(p['w2'][e],p['w2_scale'][e],format='mxfp4')
        value=(mid@down.T)*rw[rows,choices,None]
        out[rows]=out[rows]+bf16_atomic_operand_reference(value)
    return out.bfloat16()
