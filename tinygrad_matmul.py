import numpy as np
from typing import List, cast
from tinygrad.tensor import Tensor
from tinygrad.ops import Device, Compiled
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import time_linearizer
from tinygrad.graph import print_tree
from tinygrad.lazy import vars_from_ast
from tinygrad.shape.symbolic import sym_infer
from tinygrad.helpers import ansilen, getenv
from tinygrad.codegen.kernel import Opt, OptOps

def run_kernel(name, lin, rawbufs):
  lin.linearize()
  prg = cast(Compiled, Device[Device.DEFAULT]).to_program(lin)

  if getenv("DEBUG",0) > 0: 
    print(prg.prg)

  tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10)
  gflops = sym_infer(lin.info.flops, {k:k.min for k in vars_from_ast(lin.ast)})*1e-9/tm

  print(f"{name}     {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")

if __name__ == "__main__":

  A = np.random.randn(4096, 4096).astype(np.float32) 
  B = np.random.randn(4096, 4096).astype(np.float32)
  
  # the device being optimized for
  device: Compiled = Device[Device.DEFAULT]

  A_tiny = Tensor(A)
  B_tiny = Tensor(B)

  # three kernels are involved, the first two kernels are LoadOps and the last kernel is performing GEMM
  seen = set()
  out = A_tiny @ B_tiny
  sched = out.lazydata.schedule(seen)
  if getenv("DEBUG",0) > 0:
    print(f"optimizing for {Device.DEFAULT}")
    print(f"Matrix A: {A_tiny}")
    print(f"Matrix B: {B_tiny}")
    print(f"kernel schedule:")
    print_tree(sched[2].ast)

  # create buffers
  rawbufs = [device.buffer(sched[2].out.st.size(), sched[2].out.dtype)] + [device.buffer(x.st.size(), x.dtype) for x in sched[2].inputs]

  # dump kernel (no optimization)
  lin = Linearizer(sched[2].ast, device.linearizer_opts)
  run_kernel("dumb kernel             ", lin, rawbufs)

  # naive kernel & global memory coalescing
  lin = Linearizer(sched[2].ast, device.linearizer_opts)
  lin.apply_opt(Opt(op=OptOps.LOCAL, axis=0, amt=32))
  lin.apply_opt(Opt(op=OptOps.LOCAL, axis=1, amt=32))
  run_kernel("global memory coalescing", lin, rawbufs)

  # shared memory cache-blocking
