import numpy as np
from tinygrad.helpers import getenv
from tinygrad.runtime.ops_cuda import RawCUDABuffer, CUDAProgram, compile_cuda

kernel0_1 = """
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

kernel0_2 = """
__global__ void gemm(const float* A, const float* B, float* C) {
  int x = blockIdx.y;
  int y = blockIdx.x;
  float tmp = 0.0f;
  for (int i = 0; i < 4096; ++i) {
    float val0 = A[(x*4096) + i]; 
    float val1 = B[y+(i*4096)]; 
    tmp = ((val0*val1)) + tmp;
  }
  C[(x*4096)+y] = tmp;
}
"""

kernel1_1 = """
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

kernel1_2 = """
__global__ void gemm(const float* A, const float* B, float* C) {
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp = 0.0f;
  for (int i = 0; i < 4096; ++i) {
    float val0 = A[(x*4096) + i]; 
    float val1 = B[y+(i*4096)]; 
    tmp = ((val0*val1)) + tmp;
  }
  C[(x*4096)+y] = tmp;
}
"""

def run_kernel(name, kernel, local_dim, global_dim):

  N = 4096

  A = np.random.randn(4096, 4096).astype(np.float32) 
  B = np.random.randn(4096, 4096).astype(np.float32)
   
  a = RawCUDABuffer.fromCPU(A)
  b = RawCUDABuffer.fromCPU(B)
  c = RawCUDABuffer.fromCPU(np.zeros((N,N),dtype=np.float32))

  FLOPS = N*N*N*2

  if getenv("DEBUG", 0) > 0:
    print(kernel)

  prog = CUDAProgram("gemm", compile_cuda(kernel))

  tm = min([prog(global_dim, local_dim, a, b, c, wait=True) for _ in range(10)])
  print(f"{name}    {str(global_dim):18s} {str(local_dim):12s} takes {tm*1000:7.2f} ms, {FLOPS*1e-9/tm:6.0f} GFLOPS")
  
  np.testing.assert_allclose(A @ B, c.toCPU().reshape((N,N)), atol=1e-3, rtol=1e-3)

run_kernel("dumb kernel v1 ", kernel0_1, [1, 1, 1],   [4096, 4096, 1])
run_kernel("dumb kernel v2 ", kernel0_2, [1, 1, 1],   [4096, 4096, 1])
run_kernel("naive kernel v1", kernel1_1, [32, 32, 1], [128, 128, 1])
run_kernel("naive kernel v2", kernel1_2, [32, 32, 1], [128, 128, 1])
