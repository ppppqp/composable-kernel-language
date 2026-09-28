#include <cuda_runtime.h>

extern "C" __global__ void field(const float *input, float *output, int size, float bias) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= size)
    return;
  float value = input[index];
#pragma unroll 1
  for (int iteration = 0; iteration < 128; ++iteration)
    value = fmaf(value, 0.99991f, bias) + 0.00001f * __sinf(value);
  output[index] = value;
}

extern "C" __global__ void join(const float *first, const float *second, float *output, int size) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < size)
    output[index] = first[index] * 0.625f + second[index] * 0.375f;
}
