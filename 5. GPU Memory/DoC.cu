#include <stdio.h>
#include <cuda.h>

__global__ void dkernel(int degree, int* a){
    int accesses = 33 - degree;
    for(int i=0; i < accesses; i++){
        a[32*i] = 1;
    }
    return;
}
// Only writing kernel for changing degree of coalescing.