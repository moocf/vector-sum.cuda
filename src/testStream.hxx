#pragma once
#include <algorithm>
#include "_main.hxx"
#include "testMultiple.hxx"

using std::min;


// Perform vector addition in a number of small chunks,
// with a single stream.
//
// a: output vector (x + y)
// x: input vector 1
// y: input vector 2
// N: vector size (a, x, y)
// C: chunk size (C < N)
float testStream(float *a, float *x, float *y, int N, int C) {
  size_t C1 = C * sizeof(float);

  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );
  TRY( cudaEventRecord(start, 0) );

  cudaStream_t stream;
  TRY( cudaStreamCreate(&stream) );

  float *aD, *xD, *yD;
  TRY( cudaMalloc(&aD, C1) );
  TRY( cudaMalloc(&xD, C1) );
  TRY( cudaMalloc(&yD, C1) );

  for (int i=0; i<N; i+=C) {
    int c = min(C, N-i);
    size_t c1 = c * sizeof(float);

    TRY( cudaMemcpyAsync(xD, x+i, c1, cudaMemcpyHostToDevice, stream) );
    TRY( cudaMemcpyAsync(yD, y+i, c1, cudaMemcpyHostToDevice, stream) );
    kernelMultiple<<<64, 64, 0, stream>>>(aD, xD, yD, c);
    TRY( cudaMemcpyAsync(a+i, aD, c1, cudaMemcpyDeviceToHost, stream) );
  }

  float duration;
  TRY( cudaStreamSynchronize(stream) );
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaStreamDestroy(stream) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaEventDestroy(start) );
  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return duration;
}
