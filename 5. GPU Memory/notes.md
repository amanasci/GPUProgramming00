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