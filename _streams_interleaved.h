#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "support.h"
#include "_multiple.h"


// Perform vector addition in a number of small chunks,
// with two streams, in an interleaved manner, every
// two chunks.
//
// a: output vector (x + y)
// x: input vector 1
// y: input vector 2
// N: vector size (a, x, y)
// C: chunk size (C < N)
float test_streams_interleaved(float *a, float *x, float *y, int N, int C) {
  size_t C1 = C * sizeof(float);

  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );
  TRY( cudaEventRecord(start, 0) );

  cudaStream_t stream0, stream1;
  TRY( cudaStreamCreate(&stream0) );
  TRY( cudaStreamCreate(&stream1) );


  float *xD0, *yD0, *aD0;
  TRY( cudaMalloc(&xD0, C1) );
  TRY( cudaMalloc(&yD0, C1) );
  TRY( cudaMalloc(&aD0, C1) );

  float *xD1, *yD1, *aD1;
  TRY( cudaMalloc(&xD1, C1) );
  TRY( cudaMalloc(&yD1, C1) );
  TRY( cudaMalloc(&aD1, C1) );

  for (int i=0; i<N; i+=2*C) {
    int c = MIN(C, N-i);
    int d = MAX(0, MIN(C, N-i-C));
    size_t c1 = c * sizeof(int);
    size_t d1 = d * sizeof(int);

    TRY( cudaMemcpyAsync(xD0, x+i, c1, cudaMemcpyHostToDevice, stream0) );
    TRY( cudaMemcpyAsync(xD1, x+C+i, d1, cudaMemcpyHostToDevice, stream1) );

    TRY( cudaMemcpyAsync(yD0, y+i, c1, cudaMemcpyHostToDevice, stream0) );
    TRY( cudaMemcpyAsync(yD1, y+C+i, d1, cudaMemcpyHostToDevice, stream1) );

    kernel_multiple<<<64, 64, 0, stream0>>>(aD0, xD0, yD0, c);
    kernel_multiple<<<64, 64, 0, stream1>>>(aD1, xD1, yD1, d);

    TRY( cudaMemcpyAsync(a+i, aD0, c1, cudaMemcpyDeviceToHost, stream0) );
    TRY( cudaMemcpyAsync(a+C+i, aD1, d1, cudaMemcpyDeviceToHost, stream1) );
  }

  float duration;
  TRY( cudaStreamSynchronize(stream0) );
  TRY( cudaStreamSynchronize(stream1) );
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaStreamDestroy(stream0) );
  TRY( cudaStreamDestroy(stream1) );
  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaFree(aD1) );
  TRY( cudaFree(yD1) );
  TRY( cudaFree(xD1) );
  TRY( cudaFree(aD0) );
  TRY( cudaFree(yD0) );
  TRY( cudaFree(xD0) );
  return duration;
}
