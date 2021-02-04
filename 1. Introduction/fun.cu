#include <stdio.h>
#include <cuda.h>
#define N 100


__global__ void fun() {
    for(int i=0;i<N;++i){
        printf("%d\n",i*i)
    }
}

int main(){
    fun<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}