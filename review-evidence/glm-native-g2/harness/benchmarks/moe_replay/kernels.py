"""Benchmark-only activation preloading; no expert weights or scratch touched."""
import triton as tr
import triton.language as tl


@tr.jit
def touch_input(x, sink, N: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(x+i,i<N,other=0).to(tl.float32)
    # Observable output prevents removal of the loads. One word per workgroup.
    tl.store(sink+tl.program_id(0),tl.sum(v))
