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
const int M = 4096;
const int N = 4096;
const int K = 4096;
__global__ void sgemm_naive(const float *A, const float *B, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = tmp;
  }
}
"""

kernel1 = """
__global__ void sgemm_naive(const float* A, const float* B, float* C) {
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

kernel2 = """
__global__ void r_4096_4096_4096(const float* data1, const float* data2, float* data0) {
  int gidx0 = blockIdx.y; /* 4096 */
  int gidx1 = blockIdx.x; /* 4096 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4096; ++ridx0) {
    float val0 = data1[(gidx0*4096)+ridx0];
    float val1 = data2[gidx1+(ridx0*4096)];
    acc0 = ((val0*val1)+acc0);
  }
  data0[(gidx0*4096)+gidx1] = acc0;
}
"""
kernels = [kernel0, kernel1, kernel2]
for kernel in kernels:
  print(kernel)
  prog = CUDAProgram("sgemm_naive", compile_cuda(kernel))

  local_dim = [1, 1, 1]
  global_dim = [4096, 4096, 1]

  tm = min([prog(global_dim, local_dim, a, b, c, wait=True) for _ in range(10)])
  print(f"takes {tm*1000:7.2f} ms, {FLOPS*1e-9/tm:6.0f} GFLOPS matmul")

  np.testing.assert_allclose(A @ B, c.toCPU().reshape((N,N)), atol=1e-3, rtol=1e-3)

