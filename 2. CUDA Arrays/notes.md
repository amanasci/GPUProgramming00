# Important Points:

## Order:
    1. arrayfun . cu
    2. arrayfun2 . cu
    3. 


## arrayfun . cu 
1. `*da` is the pointer intialized on the CPU memory but it is actually storing the locations of GPU memory.
2. `cudaMalloc` is the function used to allocate memory on the GPU. It is same as normal Malloc in C.
3. `cudaMemcpy` actually copies the data from "device" --> GPU  to "host" --> CPU.
4. No `cudaDeviceSynchronize()` is used as cudaMemecpy also waits for the kernel to execute the program. Else there was a need of it.
5. cudaMemcpy waits as the memory it needs to copy is being used by kernel and thus it can't work before kernel finishes. Though there is an async version of cudaMemcpy.


## arrayfun . cu
1. Everything is ok until N = 1024. 
2. At N = 1025. Output is unexpected and is a string of zeroes.

## Thread Organization:
1. Kernel is always launched as grid of threads.
2. A grid is 3D array of thread-blocks (`gridDim.x`, `gridDim.y` and `gridDim.z`). Thus each block has `blockIdx.x`,`.y`,`.z`
3. A thread-block is a 3D array of threads. (`blockDim.x`,`.y`,`.z`). Thus each thread has `threadIdx.x`,`.y`,`.z`
4. This specific organisation makes working with Image Processing tasks and Solving PDEs on volumes easier.
    ### Typical configuration: 
    1. 1-5 blocks per SM (Streaming Multiprocessor)
    2. 128-1024 threads per block.
    3. Total 2k - 100k threads.
