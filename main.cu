#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>
#include "error.h"


// 1. Check how many compute devices are attached.
// 2. Select device with atleast compute capability 1.3.
int main() {
  int id;                     // 1
  TRY( cudaGetDevice(&id) );  // 1
  printf("Current CUDA device: %d\n", id);

  cudaDeviceProp p;                 // 2
  memset(&p, 0, sizeof(p));         // 2  
  p.major = 1;                      // 2
  p.minor = 3;                      // 2
  TRY( cudaChooseDevice(&id, &p) ); // 2
  printf("CUDA device with atleast compute capability 1.3: %d\n\n", id);
  printf("Cards that have compute capability 1.3 or higher\n"
         "support double-precision floating-point math.\n");
  TRY( cudaSetDevice(id) );         // 2
  return 0;
}
