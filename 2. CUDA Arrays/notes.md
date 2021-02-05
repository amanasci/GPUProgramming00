# Important Points:

## Order:
    1. arrayfun . cu
    2. arrayfun2 . cu
    3. arrayfun3 . cu
    4. accessing . cu
    5. 2Darrayfun . cu


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


## 2Darrayfun . cu:
1. `define N 5` sets constant N equals to 5. Likewise `define M 6` sets M equal to 6. So when grid is created using `dim3 block(N,M,1)` it defines a 3D block with N columns and M rows with 1 in depth i.e `threadIdx.x == N`. `.y == M` and `.z ==1`.
2. `hmatrix` is matrix in CPU's memory whereas `matrix` is matrix in GPU's memory.
3. Basically id = xindex(row) * width + yindex(column). Used to get indices of a matrix in the form an array.
4.  **Explanation of loop** : ii loop goes for the x axis of the matrix (Row). jj loop goes for the y axis of the matrix, i.e (Column). So it goes like, for every row, go through all the members of the column and print them.