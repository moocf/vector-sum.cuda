#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#ifndef TRY_CUDA
inline void try_cuda(cudaError err, const char* exp, const char* func, int line, const char* file) {
  if (err == cudaSuccess) return;
  fprintf(stderr,
    "%s: %s\n"
    "  in expression %s\n"
    "  at %s:%d in %s\n",
    cudaGetErrorName(err), cudaGetErrorString(err),
    exp,
    func, line, file);
  exit(err);
}

// Prints an error message and exits, if CUDA expression fails.
// TRY_CUDA( cudaDeviceSynchronize() );
#define TRY_CUDA(exp) try_cuda(exp, #exp, __func__, __LINE__, __FILE__)
#endif

#ifndef TRY
// Prints an error message and exits, if CUDA expression fails.
// TRY( cudaDeviceSynchronize() );
#define TRY(exp) TRY_CUDA(exp)
#endif
