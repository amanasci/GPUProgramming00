## Global Memory
1. Main means of communication data b/w host and device.
2. Long Latency 
3. Content visible to all GPU threads

### Texture Memory 
1. small (12KB)
2. Optimized for 2D spatial location
3. Read only across threads 

### Constant Memory 
1. Read only across threads
2. Small (64KB)

### L2 Cache 
1. 768KB
2. Shared among SMs
3. Fast atomics**

### L1 Cache/ Shared Memory
1. 64KB per SM
2. Low Latency 
3. High Bandwidth

### Registers 
1. 32K per SM
2. Max 21 registers per thread
3. High Bandwidth


### Bandwidth: 
1. Big data bus than fast data bus
2. Parallel data bus.
3. Technique to improve bandwidth:
    1. Share/reuse data
    2. Data Compression [Lossy and Lossless]
    3. Recompute than store + fetch

### Latency:
1. Time required for I/O
2. Latency should be minimized; ideally zero. 
   1. Processor should have data available in no time.
   2. In practice, memory I/O becomes bottleneck.
3. Latency hiding on GPUs is done by exploiting massive multi threading. While data is being loaded, use different warp to save time.


### Locality 
1. It is important in GPU's also. 
2. Types: Spatial , Temporal
3. Spatial: If `a[i]` is accessed, `a[i+k]` is also accessed.
4. Temporal: If `a[i]` is accessed now it'll again be accessed soon.

### Memory Coalescing
1. If warp threads access words from the same block of 32 words, their memeory requests are club into one. 
2. That is memory requests are `coalesced`.
3. Without coalescing, each `load/store` instruction would require one memory cycle. Too much time will be needed.

### Degree of Coalescing
1. DoC is the `inverse (33 - {} )` of the number of memory transactions required for a warp to execute an instruction.

### Shared Memory
1. Programmable L1 cache / scratchpad memory.
2. Accessible only in thread block.
3. Useful for repoeated small data or coordination.
4.  ```__shared__ float a[N];```, ```__shared__ unsigned s;``` this how we init memory in shared memory.
5.  ```__syncthreads()``` is the barrier. It makes kernel wait for a thread to execute in a thread block.


### sharedmem . cu and sharedmembarrier . cu
1. There is difference in the two codes. In first we can get output as either 3 or 1 as the `threadIdx.x = 0`, `1` are in different warp than `10`. So They are not in sync. 
2. While in second we are using barrier to force the sequentiality between different warps. So output is always 3.
3. Do we need all `__synthreads()` used in code? Not really. We don't really need first `__syncthreads()` but we need second and third. 
4. Why second? Third is understandable. Second is needed as if there is no sync then there's chance that both the if conditions above and below second one will execute at same time which might not really be a good thing for us. 
5. Cost of ```__syncthreads()``` is not high.