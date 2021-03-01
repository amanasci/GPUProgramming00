# Important Points:

## Order:
    1. generalconfig . cu 



## generalconfig . cu
1.  We take a simple approach for huge data. We will use 1D config for this. 
2.  Since thread limit in each block is 1024, we fix the `BLOCKSIZE` at 1024.
3.  Number of blocks to be launched will be calculated based on data size. 
4.  Then kernel will check for the number of data points and use only required number of threads to write.


## GPU Computation Hiearchy [At hardware level]
1. Thread - 1 thread
2. Warp (group of threads) - 32 threads [A GPU always launches Warps at hardware level. To execute single thread, 31 threads are masked out of launched Warp.] [ They excute in SIMD fashion, SIMD == Single instruction multiple data ]
3. Block - 1024 threads
4. Multi Processor - Tens of thousands of threads
5. GPU - Hundreds of thousands of threads
> Note: Grid is a software concept it does not exist at hardware level.