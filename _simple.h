#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "support.h"


// Each thread computes the sum of a single component of vector.
// We must also ensure to not access out of bounds.
//
// threadIdx.x: thread index, within block
// blockIdx.x:  block index, within grid
// blockDim.x:  number of threads in a block
// i: index into the vectors
__global__ void kernel_simple(float *a, float *x, float *y, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) a[i] = x[i] + y[i];
}


// 1. Allocate space for A, X, Y on GPU.
// 2. Copy X, Y from host memory to device memory (GPU).
// 3. Execute kernel with 256 threads per block.
// 4. Wait for kernel to complete, and copy A from device to host memory.
// 5. Free A, X, Y allocated on GPU.
float run_simple(float *a, float *x, float *y, int N) {
  size_t N1 = N * sizeof(float);
  
  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );
  TRY( cudaEventRecord(start, 0) );

  float *aD, *xD, *yD;
  TRY( cudaMalloc(&aD, N1) ); // 1
  TRY( cudaMalloc(&xD, N1) ); // 1
  TRY( cudaMalloc(&yD, N1) ); // 1
  TRY( cudaMemcpy(xD, x, N1, cudaMemcpyHostToDevice) ); // 2
  TRY( cudaMemcpy(yD, y, N1, cudaMemcpyHostToDevice) ); // 2

  int threads = 256;                                 // 3
  int blocks  = CEILDIV(N, threads);                 // 3
  kernel_simple<<<blocks, threads>>>(aD, xD, yD, N); // 3

  float duration;
  TRY( cudaMemcpy(a, aD, N1, cudaMemcpyDeviceToHost) ); // 4
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaFree(yD) ); // 5
  TRY( cudaFree(xD) ); // 5
  TRY( cudaFree(aD) ); // 5
  return duration;
}
