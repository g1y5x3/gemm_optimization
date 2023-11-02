import numpy as np
from tinygrad.runtime.ops_cuda import RawCUDABuffer, CUDAProgram, compile_cuda

N = 4096

A = np.random.randn(4096, 4096).astype(np.float32) 
B = np.random.randn(4096, 4096).astype(np.float32)
 
a = RawCUDABuffer.fromCPU(A)
b = RawCUDABuffer.fromCPU(B)
c = RawCUDABuffer.fromCPU(np.zeros((N,N),dtype=np.float32))

FLOPS = N*N*N*2

kernel0 = """
__global__ void gemm(const float* A, const float* B, float* C) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  float tmp = 0.0f;
  for (int i = 0; i < 4096; ++i) {
    float val0 = A[(x*4096) + i]; 
    float val1 = B[y+(i*4096)]; 
    tmp = ((val0*val1)) + tmp;
  }
  C[(x*4096)+y] = tmp;
}
"""

kernel1 = """
__global__ void gemm(const float* A, const float* B, float* C) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float tmp = 0.0f;
  for (int i = 0; i < 4096; ++i) {
    float val0 = A[(x*4096) + i]; 
    float val1 = B[y+(i*4096)]; 
    tmp = ((val0*val1)) + tmp;
  }
  C[(x*4096)+y] = tmp;
}
"""

print(kernel0)
prog = CUDAProgram("gemm", compile_cuda(kernel0))

local_dim = [1, 1, 1]
global_dim = [4096, 4096, 1]

tm = min([prog(global_dim, local_dim, a, b, c, wait=True) for _ in range(10)])
print(f"dumb kernel      {str(global_dim):18s} {str(local_dim):12s} takes {tm*1000:7.2f} ms, {FLOPS*1e-9/tm:6.0f} GFLOPS matmul")

np.testing.assert_allclose(A @ B, c.toCPU().reshape((N,N)), atol=1e-3, rtol=1e-3)

print(kernel1)
prog = CUDAProgram("gemm", compile_cuda(kernel1))

local_dim = [32, 32, 1]
global_dim = [128, 128, 1]

tm = min([prog(global_dim, local_dim, a, b, c, wait=True) for _ in range(10)])
print(f"naive kernel     {str(global_dim):18s} {str(local_dim):12s} takes {tm*1000:7.2f} ms, {FLOPS*1e-9/tm:6.0f} GFLOPS matmul")

np.testing.assert_allclose(A @ B, c.toCPU().reshape((N,N)), atol=1e-3, rtol=1e-3)

