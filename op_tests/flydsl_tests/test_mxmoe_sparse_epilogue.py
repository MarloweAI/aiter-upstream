# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Marlowe AI. All rights reserved.
"""Native G2 fixture with exact positive expert sums, padding holes and graph reuse.

Run on gfx950 with the repository's FlyDSL dependencies installed:
    python op_tests/flydsl_tests/test_mxmoe_sparse_epilogue.py

This isolates the epilogue's row/route contract. The adoption campaign additionally
checks the full native sort + G1 + G2 operator on captured model weights/inputs.
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2


def _fixture(M, holes):
    # E2M1 positive nibble codes 1..4 represent .5, 1, 1.5 and 2. Every
    # expert's constant packed W and constant E8M0 scale remain invariant
    # under the native packed permutations; no second matrix library is needed.
    values = (0.5, 1.0, 1.5, 2.0)
    routed = {}
    expected = torch.zeros((M, 6144), dtype=torch.float32)
    for token in range(M):
        for slot in range(9):
            expert = (token + slot * 31) % 256 if slot < 8 else 256
            weight = 0.0625 if slot < 8 else 1.0
            routed.setdefault(expert, []).append(((slot << 24) | token, weight))
            expected[token] += 2.0 * values[expert % 4] * weight
    ids, weights, experts = [], [], []
    for expert, entries in sorted(routed.items()):
        for first in range(0, len(entries), 16):
            chunk = entries[first : first + 16]
            rows = [(M, 0.0)] * 16
            # A valid row may be either member of a wave's pair, in either MR.
            positions = list(range(15, -1, -1)) if holes else list(range(16))
            for entry, pos in zip(chunk, positions):
                rows[pos] = entry
            ids.extend(row[0] for row in rows)
            weights.extend(row[1] for row in rows)
            experts.append(expert)
    total = len(ids)
    device = "cuda"
    w = torch.empty((257, 6144, 128), dtype=torch.uint8, device=device)
    for expert in range(257):
        nibble = expert % 4 + 1
        w[expert].fill_(nibble | nibble << 4)
    inputs = dict(
        inter_sorted_quant=torch.full(
            (total, 128), 0x22, dtype=torch.uint8, device=device
        ),
        inter_sorted_shuffled_scale=torch.full(
            (total // 16, 256), 120, dtype=torch.uint8, device=device
        ),
        w2_u8=w,
        w2_scale_u8=torch.full((257, 6144, 8), 127, dtype=torch.uint8, device=device),
        sorted_expert_ids=torch.tensor(experts, dtype=torch.int32, device=device),
        cumsum_tensor=torch.tensor([total, M * 9], dtype=torch.int32, device=device),
        sorted_token_ids=torch.tensor(ids, dtype=torch.int32, device=device),
        sorted_weights=torch.tensor(weights, dtype=torch.float32, device=device),
        M_logical=M,
        max_sorted=total,
        NE=257,
        D_HIDDEN=6144,
        D_INTER=256,
        topk=9,
        BM=16,
        BN=128,
        BK=128,
        SBM=16,
        g2_bf16_lds=True,
    )
    return inputs, expected.bfloat16().to(device)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 required")
@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("holes", [False, True])
def test_native_epilogue_padding_shared_routes_and_graph_reuse(M, holes, monkeypatch):
    monkeypatch.setenv("MXFP4_G2_KSTATIC", "1")
    args, expected = _fixture(M, holes)
    immutable = {
        name: value.clone()
        for name, value in args.items()
        if isinstance(value, torch.Tensor)
    }
    outputs = [torch.empty_like(expected), torch.empty_like(expected)]

    def run(enabled, output):
        output.zero_()  # Each invocation owns native atomic accumulation reset.
        return mxfp4_moe_gemm2(**args, out=output, g2_skip_padded_lds=enabled)

    for enabled, output in zip((False, True), outputs):
        for _ in range(3):
            run(enabled, output)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(enabled, output)
        for _ in range(3):
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    for name, before in immutable.items():
        assert torch.equal(
            args[name], before
        ), f"immutable native input changed: {name}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
