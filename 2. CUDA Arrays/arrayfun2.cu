#include <stdio.h>
#include <cuda.h>


__global__ void init(int *a, int alen) {
    unsigned id = threadIdx.x;
    if(id<alen) a[id] = 0;
}
 int main(){
    int *da, N;
    N=32;
    cudaMalloc(&da, N * sizeof(int));
    
    init<<<1, N>>>(da, N);
    cudaDeviceSynchronize();

    int a[N];
    cudaMemcpy(a, da, N* sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<N;++i){
        printf("%d ",a[i]);
    }
    printf("\n");
    return 0;
 }