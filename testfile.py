import numpy as np
import cupy as cp

from cupyx.profiler import benchmark

# print(benchmark(squared_diff, (x, y), n_repeat=1000))

loaded_from_source = r'''
extern "C"
{
__global__ void test_sum(const float* x1, int* y, const unsigned int N, const int c)
   {
   unsigned int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
      if (tid < N * N)
      {
         if(x1[tid] == 0)
         {
            y[tid] = x1[tid] + c;
         }
      }
   }
}
'''

module = cp.RawModule(code=loaded_from_source)
ker_sum = module.get_function('test_sum')
N = 10
x1 = cp.zeros(N**2, dtype=cp.float32).reshape(N, N)
# x2 = cp.ones((N, N), dtype=cp.float32)
y = cp.zeros((N, N), dtype=cp.uint32)
ker_sum((N,), (N,), (x1, y, N, 5))
print(x1)
print(y)
