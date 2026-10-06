#include <hip/hip_runtime.h>

extern "C" __global__ void add_bias(float *values, int elements, float bias) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < elements)
    values[index] += bias;
}
