import os
import numpy as np
os.environ["CUDA"] = "1"
from tinygrad.runtime.ops_cuda import RawCUDABuffer, CUDAProgram, compile_cuda

FLOAT16 = True
ACC_FLOAT16 = False
N = 4096

na = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32)

if FLOAT16:
  na = na.astype(np.float16)
  nb = nb.astype(np.float16)

a = RawCUDABuffer.fromCPU(na)
b = RawCUDABuffer.fromCPU(nb)
c = RawCUDABuffer.fromCPU(np.ones((N,N),dtype=np.float32))

FLOPS = N*N*N*2
BW = N*N*3*4

prog = CUDAProgram("wmma_example", compile_cuda(f"""
#include <mma.h>
using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half *a, half *b, float *c)
{{
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < {N}; i += WMMA_K) {{
    int aRow = warpM * WMMA_M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a + aRow + aCol * {N}, {N});
    wmma::load_matrix_sync(b_frag, b + bRow + bCol * {N}, {N});

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }}

  // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  // Store the output
  wmma::store_matrix_sync(c + cRow + cCol * {N}, c_frag, {N}, wmma::mem_col_major);
}}
"""))

# local_dim[0] must be a multiple of warpSize(32)
# 128x4 means we have 16 warps and a block computes a 64x64 output tile
local_dim = [128, 4, 1]

# the 16 used in global_dim[0] and global_dim[1] correspond to WMMA_M, WMMA_N
global_dim = [int((N+16*local_dim[0]/32-1)/(16*local_dim[0]/32)), int((N+16*local_dim[1]-1)/(16*local_dim[1])), 1]

tm = min([prog(global_dim, local_dim, a, b, c, wait=True) for _ in range(20)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")

np.testing.assert_allclose(na.T.astype(np.float32) @ nb.T.astype(np.float32), c.toCPU().reshape((N,N)).T, atol=1e-2, rtol=1e-2)
