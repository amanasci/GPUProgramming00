#include <stdio.h>
#include <cuda.h>


__global__ void init(int *a, int alen) {
    unsigned id = threadIdx.x;
    if(id<alen) a[id] = 0;
}


__global__ void add(int *a, int alen) {
    unsigned id = threadIdx.x;
    if(id<alen) a[id] = id;
}


 int main(){
    int *da, N;
    N=1024;
    cudaMalloc(&da, N * sizeof(int));
    
    init<<<1, N>>>(da, N);
    add<<<1, N>>>(da,N);
    cudaDeviceSynchronize();

    int a[N];
    cudaMemcpy(a, da, N* sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<N;++i){
        printf("%d ",a[i]);ss
    }
    printf("\n");
    return 0;
 }