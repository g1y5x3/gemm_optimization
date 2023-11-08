import numpy as np
from tinygrad.helpers import getenv
from tinygrad.runtime.ops_cuda import RawCUDABuffer, CUDAProgram, compile_cuda

kernel0 = """
__global__ void gemm_dumb(const float* A, const float* B, float* C) {
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

kernel1 = """
__global__ void gemm_naive(const float* A, const float* B, float* C) {
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

kernel2_0 = """
__global__ void r_128_128_32_32_4096(const float* data1, const float* data2, float* data0) {
  int gidx0 = blockIdx.y; /* 128 */
  int gidx1 = blockIdx.x; /* 128 */
  int lidx2 = threadIdx.y; /* 32 */
  int lidx3 = threadIdx.x; /* 32 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4096; ++ridx0) {
    float val0 = data1[(gidx0*131072)+(lidx2*4096)+ridx0];
    float val1 = data2[(gidx1*32)+lidx3+(ridx0*4096)];
    acc0 = ((val0*val1)+acc0);
  }
  data0[(gidx0*131072)+(gidx1*32)+(lidx2*4096)+lidx3] = acc0;
}
"""

kernel2_1 = """
__global__ void gemm_global_mem_coalesce(const float *A, const float *B, float *C) {
  const int cRow = blockIdx.x * 32 + (threadIdx.x / 32);
  const int cCol = blockIdx.y * 32 + (threadIdx.x % 32);
  float tmp = 0.0;
  for (int i = 0; i < 4096; ++i) {
    float val0 = A[cRow * 4096 + i];
    float val1 = B[i * 4096 + cCol];
    tmp = val0*val1 + tmp;
  }
  C[cRow * 4096 + cCol] = tmp;
}
"""

kernel2_2 = """
__global__ void gemm_global_mem_coalesce(const float* A, const float* B, float* C) {
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

kernel3_2 = """
__global__ void sgemm_shared_mem_block(const float *A, const float *B, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[32 * 32];
  __shared__ float Bs[32 * 32];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x;
  const uint threadRow = threadIdx.y;

  // advance pointers to the starting positions
  A += cRow * 32 * 4096;                // row=cRow, col=0
  B += cCol * 32;                       // row=0, col=cCol
  C += cRow * 32 * 4096 + cCol * 32;    // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < 4096; bkIdx += 32) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * 32 + threadCol] = A[threadRow * 4096 + threadCol];
    Bs[threadRow * 32 + threadCol] = B[threadRow * 4096 + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += 32;
    B += 32 * 4096;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < 32; ++dotIdx) {
      tmp += As[threadRow * 32 + dotIdx] * Bs[dotIdx * 32 + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * 4096 + threadCol] = tmp;
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

  tm = min([prog(a, b, c, global_size=global_dim, local_size=local_dim, wait=True) for _ in range(10)])
  print(f"{name}    {str(global_dim):18s} {str(local_dim):12s} takes {tm*1000:7.2f} ms, {FLOPS*1e-9/tm:6.0f} GFLOPS")
  
  np.testing.assert_allclose(A @ B, c.toCPU().reshape((N,N)), atol=1e-3, rtol=1e-3)

run_kernel("dumb kernel                  ", kernel0,   [1, 1, 1], [4096, 4096, 1])
run_kernel("naive kernel                 ", kernel1,   [32, 32, 1], [128, 128, 1])
run_kernel("global memory coalescing v0  ", kernel2_0, [32, 32, 1], [128, 128, 1])
run_kernel("global memory coalescing v1  ", kernel2_1, [1024, 1, 1], [128, 128, 1])
run_kernel("global memory coalescing v2  ", kernel2_2, [32, 32, 1], [128, 128, 1])
# run_kernel("shared memory memory block v1", kernel3_1, [1024, 1, 1], [128, 128, 1])
# run_kernel("shared memory memory block v2", kernel3_2, [32, 32, 1], [128, 128, 1])
