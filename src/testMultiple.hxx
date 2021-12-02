#pragma once
#include <algorithm>
#include "_main.hxx"

using std::max;


// Each thread can compute the sum of multiple components of vectors. Each
// thread computes the sum of its respective component, and shifts by a
// stride of the total number of vectors. This is done as long as it does
// not exceed the length of the vectors.
//
// 1. Compute sum at respective index, while within bounds.
// 2. Shift to the next component, by a stride of total no. of threads.
//
// threadIdx.x: thread index, within block (0 ... 1)
// blockIdx.x:  block index, within grid (0 ... 1)
// blockDim.x:  number of threads in a block (2)
// i: index into the vectors
__global__ void kernelMultiple(float *a, float *x, float *y, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // 1
  while (i < N) {                                // 1
    a[i] = x[i] + y[i];                          // 1
    i += gridDim.x * blockDim.x; // 2
  }
}


// 1. Allocate space for A, X, Y on GPU.
// 2. Copy X, Y from host memory to device memory (GPU).
// 3. Execute kernel with 256 threads per block, and max 4 blocks.
// 4. Wait for kernel to complete, and copy A from device to host memory.
// 5. Free A, X, Y allocated on GPU.
//
// a: output vector (x + y)
// x: input vector 1
// y: input vector 2
// N: vector size (a, x, y)
float testMultiple(float *a, float *x, float *y, int N) {
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

  int threads = 256;                                   // 3
  int blocks  = max(ceilDiv(N, threads), 4);           // 3
  kernelMultiple<<<blocks, threads>>>(aD, xD, yD, N); // 3

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
