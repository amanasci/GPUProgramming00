#include <stdio.h>
#include <cuda.h>

#define BLOCKSIZ 1024


__global__ void dkernel() {
    __shared__ unsigned s;

    if(threadIdx.x == 0) s=0;
    if(threadIdx.x == 1) s+=1;
    if(threadIdx.x == 100) s+=2;
    if(threadIdx.x == 0) printf("s = %d\n",s);
}

int main(){
    int i;
    for (i =0, i<10, ++i){
        dkernel<<2, BLOCKSIZ>>();
        cudaDeviceSynchronize();

    }
    return 0;
}