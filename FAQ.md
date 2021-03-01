### What is clock frequency?
> It is the speed at which a processor / core runs. For instance, if your laptop is “Intel(R) Core(TM) i5-2467M CPU @ 1.60GHz”. This means each core on this laptop runs at 1.6 * 109 cycles per second. Typically, higher the clock frequency, faster is the processor. For GPUs, this clock is likely to be slower.


### While explaining why many levels of cache reasons, please explain size v/s cost, spatial and temporal reference
> Say there are three caches L1, L2, L3, where L1 is closest to the core, and L3 is closest to the DRAM. Since L1 is closest to the core, it needs to be very fast. Therefore, it is costly. Hence, its size is typically small. L3 bridges the gap between L2 and DRAM. Hence, it is relatively slower (but faster than DRAM). Hence it is cheaper (due to low-end technology requirement). Therefore, it is usually larger in size. Here, DRAM means main memory.

> A typical access pattern is: ```for (i = 0; i < N; ++i) a[i] = 0;``` Here, as soon as we access ```a[0]```, we are also going to access `a[1]`. Similarly, `a[2]`, etc. Thus, there is a locality of reference in “space”. That is, spatial locality exists in this access pattern (accessing `a[x]`, accesses `a[x+1]` also).

> Temporal locality means if x is accessed now, it would again be accessed again soon. Thus, locality of reference exists in “time”. For instance, variable i in the above for loop has temporal locality.


### CPU function, GPU kernel run asynchronously what does it mean..?
> In our usual C or Fortran program, when we call a function foo from main, the function main waits till foo finishes. In case of CUDA/GPU programs, when main calls a kernel on GPU, main does not wait by default. It continues executing the next instruction after the kernel launch. Thus, CPU and GPU are executing their own instructions simultaneously. This is asynchronous processing.
> We can make it similar to synchronous processing by adding ```cudaDeviceSynchronize``` after every kernel call.


### A thread my jump from one core to another what does it mean?
> Say there are 4 cores on a machine. A user can launch 8 or 100 threads on this machine. However, only 4 threads can be running at any point in time (sat t1 to t4), making the remaining 96 threads wait for their turn. Now, let’s say, one thread t2 completed execution. Thread scheduler can decide to schedule t90 to this free core (second core). Thus, threads t1, t90, t3, t4 are running now. They typically run only for some time, so that other threads get a chance. Hence, the scheduler may decide to run t11, t12, t40, t95 for some time, making t1, t90, t3, t4 wait for their second turn. At this stage, let’s say t11 finishes. The scheduler may decide to run t90 on this core (core 1). Thus, t90 has jumped from core 2 to core 1. Depending upon how many times threads get rescheduled, they may jump multiple times. 


### We can launch more threads than number of cores but it may not be a good idea to launch too many threads on CPU why? Why is it ok on GPU?
> The main reason is CPUs are designed for a few threads, while GPUs are designed for large number of threads.
> Further, CPUs are not executing in SIMD fashion. Each thread is independent, so hardware has to track each one separately. In contrast, GPUs execute in SIMD fashion for the warp. Hence, less context is required.


### What is context switch overhead?
> Say, thread t1 has statements S1; S2; S3; If it is executing on a core c1, after some time, it may have finished executing S1 but not S2. At this stage, thread t2 may start running on c1. After some more time, t1 may come back on c1. At this stage, t1 should execute starting from S2 and not S1. Also, all the variables should have the same data as before in t1. This demands storing the “context” of thread t1. When a core moves from t1 to t2, it needs to store this context somewhere, so that when t1 comes back, this context can be utilized. Note that, t1’s context is getting stored, and t2’s context is getting loaded. This whole process is context-switch overhead.


### What is thread migration?
> This is the same as a thread jumping from one core to another.


### What is warp based execution and warp scheduling?
> Think about an army march. All the soldiers are walking together in exactly the same manner. This is like warp-based execution. All 32 threads in a warp execute the same instruction. On a CPU, one thread may be executing a printf statement, another may be opening a file, another may be making a multiplication. On a GPU, within a warp, all the threads would be executing exactly the same instruction (or sitting idle). But it would never happen that two threads in a warp are executing different instructions.

> Scheduling warps on multi-processor (SM) is similar to scheduling threads on cores. Thus, two warps w1 and w2 may be assigned to SM1 and SM2. Since there are several warps scheduled, w1/w2 may be replaced by w5/w90. How to replace that is decided by the warp scheduling algorithm (which is hardware specific).


### What is #define option? Any other options like it? Why do we use __device__ keyword any other keywords like this?
> This is a preprocessing macro in C/C++/CUDA. 
> ```#define XYZ 12345``` means that all the occurrences of XYZ are replaced by 12345. Think about XYZ as a variable having fixed value 12345.
> Others are`#ifdef`, `#endif`, `#if`, `#include`, `#pragma`.
> We use ```__device__``` keyword to identify functions running on GPU. Without this keyword, the function runs on the CPU. Other keywords are ```__host__```, ```__global__```. host functions run only on CPU. __global__ functions are kernels, which run only on GPU. device functions are not kernels, hence cannot be called from main.


###  What does cudaDeviceSynchronize do and why is it not required sometimes?
> Since CPU and GPU run asynchronously (concurrently or in parallel), when we want the CPU to wait for the kernel to finish, we use ```cudaDeviceSynchronize```. It makes sure that the kernel has completed execution.

>For instance, consider a scenario where we want to measure the time taken to execute a kernel. For this, we can use clock API from C library. The code would be
> ```start = clock();  kernel<<<...>>>(...);  end = clock();```
> The above code is flawed. It does not measure the kernel execution time, but only the kernel launch time. To make sure we measure the kernel execution time, we need to use CDS.
> ```start = clock();  kernel<<<...>>>(...);  cudaDeviceSynchronize(); end = clock();```
> In some cases, it is not required, because some other function waits for kernel to be over. For instance, cudaMemcpy makes sure that the previous kernel, if any, completes execution. Thus, the following is okay, which measures kernel time + cudamemcpy time.
> ```start = clock();  kernel<<<...>>>(...);  cudaMemcpy(...); end = clock();```
> In the above code, cudaMemcpy does not execute until kernel is terminated.


### Why pass by value to kernel arguments is ok and why there is a problem while pass by reference?
> This is because a kernel is running on the GPU, and it can access variables only from GPU memory. Similarly, for CPU functions.
> Consider the following code.
```c
__global__ void kernel(int y) {...}
int x=5; kernel<<<...>>>(x);
```
> Here, the value of x is copied from CPU memory to GPU memory y. But x is accessed only on CPU and y is accessed only on GPU. So this program is okay. Now with pass by reference or pointer:
```c 
__global__ void kernel(int *y) {...}
int x=5; kernel<<<...>>>(&x);
```
> Variable y is in GPU memory. But it is a pointer, pointing to x’s address, which is in CPU. Therefore, when we access y in kernel, it is alright. But when we access *y in kernel, it would result in an error. This is a wrong program.


### How thread blocks are assigned to SM’s? How exactly assigning threads to the GPU takes place?
> This is not documented, and left to NVIDIA. But a thread-block does not jump from one SM to another. Within a block, warps are assigned to SM’s cores, which executes the warps. Warps are running in time-multiplexed manner.

> What’s the hierarchy in GPU like SM>core…….?
> Hardware: GPU > SM > Core
> Hardware + Software: GPU > SM > Block > Warp > Core


### One kernel🡪One grid (1 to 1) mapping what does it mean?
> Don’t worry about this too much. Grid is simply how we number threads and blocks. Since one kernel is associated with a set of threads and blocks, all of them together is called the grid. This is to say that the same thread cannot execute another kernel’s code. Thus, if kernel1 and kernel2 are executing, they have two different sets of threads (or grids). Their numbers may be the same (e.g., thread 0), but the threads are really different. 


### Can we access 2D matrix without converting it to 1D in GPU kernel, why?
> It is not possible in general, because arrays are passed by reference. But using unified memory or host memory, you would be able to achieve this, since both CPU and GPU can access the same memory location.


### Why cudaMemcpy – blocking for cpu,gpu? why cudaMemcpyAsym-not blocking for cpu? What does these both sentences mean?
> Blocking for a device means that device cannot execute the next instruction until this function completes. Consider this code:
> ```f1(); kernel<<<...>>>(); f2(); cudaMemcpy(...); f3();```
> cudaMemcpy blocks for CPU means that f3() cannot execute until cudaMemcpy is over.
> cudaMemcpy blocks for GPU means that cudaMemcpy cannot start until kernel is over, which is running on the GPU.
> Now let’s replace cudaMemcpy with cudaMemcpyAsync.
> ```f1(); kernel<<<...>>>(); f2(); cudaMemcpyAsync(...); f3();```
> Now, cudaMemcpyAsync blocks for GPU means it cannot start until kernel is over. However, if is NOT blocking for CPU means that f3() can start even if cudaMemcpyAsync is not over.

### What is SIMD execution?
> SIMD stands for Single Instruction Multiple Data. In the context of GPUs, warp threads (32 in number) execute the same instruction, but may operate on different data items.
> For instance, if an instruction wants to load data arr[threadIdx.x] from memory, all the warp-threads execute the same load instruction, but they all load different data items arr[0]..arr[31].
> In contrast, two warps may be executing different instructions. For instance, one may be loading data while another may be performing an assignment.

### What’s difference between data parallelism and instruction level parallelism, what does it have to do in the context of cpu and gpu…….? 
> Consider the code fragment below, which is part of a for loop.
```
a[i] = 0;
b[i] = 1;
c[i] = a[i] + b[i];
d[i] = a[i] + 2;
e[i] = c[i] - d[i];
```
> ILP tries to extract parallelism among instructions. Thus, instructions 1 and 2 can be executed in parallel. Similarly, instructions 3 and 4 can be executed in parallel, but they both need to wait for instructions 1 and 2 to be over. Instruction 5 needs to wait for both 3 and 4 to be over. This is what traditional compilers tried to extract from sequential programs.
> Data parallelism, in contrast, does not care about parallelism across instructions. It assigns different data items to different threads. Thus, a[i] are assigned to different threads. Thus, all the threads execute instruction 1 together. Then all the threads execute instruction 2, and so on. Note that ILP is limited, but DP can have more and more parallelism as size of the for loop increases. Thus, DP is more scalable, and suitable for GPUs.

### What does scalable mean..?
> Scalable means that “something works” for larger “something else”. In our case, we talk about performance and number of threads. In many cases, we say performance improves with the data size. An algorithm is non-scalable if it does not provide benefits beyond a point (beyond certain data sizes, for instance).

### There are also interfaces for Python→CUDA, Javascript → OpenCL , LLVM → PTX what does it exactly mean……?
> That means there are converters or translators available to convert from one language to another. Thus, you can write your code in Python (in a certain way), and then run a translator which can produce a CUDA code corresponding to the Python code.

### What is dynamic  voltage frequency scaling..?
> Cores run at a fixed clock frequency. This frequency can be changed in the latest hardware. Typically, higher the frequency, faster is the instruction execution, but more is the power consumption. Hence, one needs to judiciously increase the frequency to improve execution time of certain instructions and reduce it for others to improve the power consumption. Typically, if a lot of number-crunching instructions are present (e.g., add, multiply), it is useful to increase frequency, while a lot of memory or synchronization instructions would benefit from reduced frequency (e.g., load, store, atomics).

### What’s compact capability..?
> It is compute capability. It defines what all operations can be done by the GPU. This is similar to asking about a standard a student is in. If she is in third standard, she may be able to do basic addition, but not division. When a student is in fifth standard, she can perform division, but would not know about integration. GPUs with CC less than 2.0 did not have atomic instructions. GPUs with CC >= 3.5 support dynamic parallelism.

### Warp-threads are fully synchronized. There is an implicit barrier after each step / instruction, can you please explain it rigorously…?
> Consider kernel statements ```a[threadIdx.x] = 0; b[threadIdx.x] = 1;```
> Consider two threads, thread 0 and 100. They belong to different warps. Hence it is possible that thread 0 is executing the first statement, while thread 100 is executing the second statement at the same time. Thus, they are not synchronized. In contrast, consider thread 0 and 31 which belong to the same warp. It can never happen that one is executing the first instruction and another is executing the other. They both would execute the first statement fully. Then they both would execute the second statement fully.
> In general, if statements are S1; S2; S3; …, Sn, after every statement, all warp-threads meet. It never happens that thread 0 and thread 10 are executing different Si and Sj. Thus, they are fully synchronized. One way to imagine this is to “assume” that there is a barrier after every Si for the warp-threads.

### what exactly is thread privatization, when does it come into picture…?
> Consider a kernel which uses `a[threadIdx.x]` in multiple statements. For instance,
```c
a[threadIdx.x] = f();
g(a[threadIdx.x]);
a[threadIdx.x] = h(a[threadIdx.x]);
```
> In this kernel, array a is shared across all the threads, but element `a[threadIdx.x]` is distinct for each thread. Thus, a compiler may decide to create a local temporary copy of `a[threadIdx.x]` in each thread and use it as below.
```c
int priv = f();
g(priv);
priv = h(priv);
a[threadIdx.x] = priv;
```
> This allows the compiler to make further optimizations local to each thread. For instance, now variable priv may be pushed to a register.
> Contrast this with a kernel that performs the following computation:
```sum += a[threadIdx.x];```
> Here, privatizing `a[threadIdx.x]` does not help, as it is not used often in the kernel.
> Further, privatizing sum would not help, as it is a read-write shared variable.

### Hardware takes care of adding NOP’s what does it mean..?
> Consider a conditional statement: ```if (threadIdx.x < 10) S1; else S2;```
> Since warp-threads execute in lock-step fashion, all would evaluate the condition threadIdx.x < 10 together. This condition is satisfied by only ten threads: 0..9. These ten threads should execute S1, while the remaining warp-threads (10..31) should execute S2. But since they all belong to the same warp, they must execute the same instruction at the same time!
> This gets solved by the hardware that the threads would either execute the same instruction or some of those would remain idle -- that is, they will not execute any useful instruction. Another way to say this is that the idle threads execute NOP (no-op for no operation). Thus, initially, 0..9 threads execute S1. At this time, due to SIMD nature, threads 10..31 execute NOP. Note that they should not execute S1, else the code output would be wrong. Once S1 is over, the warp executes S2. But only threads 10..31 execute S2, threads 0..9 execute NOP (sit idle).
> While writing the code, as a programmer, we never asked threads to execute NOPs. This is internally handled by the hardware during execution.


### Why simple if conditions are not as much a problem as for loops with different number of iterations for different warp threads…?
> Both lead to thread-divergence. But if conditions are executed only once. Loops typically are executed multiple times. Hence, the performance degradation due to thread-divergence is more in loops.


### “A thread block is a set of concurrently executing threads that can cooperate among themselves through barrier synchronization and shared memory” what’s barrier sync here and how thread cooperation among threads in a block different from cooperation among threads from different blocks, is it warp concept your trying to explain here…?
> This has nothing to do with warps. The barrier synchronization here is __syncthreads(). Using that function call, all the threads within the block synchronize.
> This is different because across blocks there is no easy way to synchronize. Further, blocks have shared memory, which allows fast communication from one thread to another within a block. In contrast, across blocks, one needs to use slow global memory.
> For instance, if the executed code is:
```c
if (threadIdx.x == 0) work = f();
__syncthreads();
if (threadIdx.x == 100) printf(“%d\n”, work);
```
> Here, threads 0 and 100 can communicate via variable work. This is possible because the two threads synchronize using __synchthreads between the write and the read. In contrast, consider the following code:
```c
if (blockIdx.x == 0 && threadIdx.x == 0) work = f();
__syncthreads();
if (blockIdx.x == 100 && threadIdx.x == 0) printf(“%d\n”, work);
```
> Here, there is no guarantee that thread 0 of block 100 would see the value written by thread 0 in block 0 -- it may see the value or it may see the old value. This is because, due to __synchthreads(), all threads in block 0 synchronize. Further, all threads in block 100 synchronize. But threads in block 0 do not synchronize with threads in block 100. They execute totally independently.
> Think about a warp as a tourist bus taking students from IIT Madras to Mahabalipuram. All the students within the bus would be at the same place on the road all the time. A block is like multiple buses starting from IIT Madras to Mahabalipuram around the same time. They will be at different junctions, different roads, but would be nearby. They can synchronize for breakfast in a hotel on the OMR, similar to __syncthreads().
> Multiple thread-blocks are similar to buses starting at different times of the day (morning, afternoon, evening). These buses could be very far apart from each other, and cannot synchronize along the road. 


### what is texture memory , constant memory ,what does Read only mean ..?please explain briefly
> These are additional memories present on the GPUs, which can be used to improve performance. Read-only means that we cannot write to variables in these memories from the kernel.
> For global memory, we can write using assignment instructions: sum = 0;
> However, if a variable is in texture memory or constant memory, one cannot write to it front he kernel (it needs to be written to from the CPU). By making variables read-only, it is guaranteed that the kernel would not modify those, hence those can be cached more effectively, leading to improved performance.


### in L2 cache properties what does fast atomics mean…?
> L2 is shared across all the cores. Similarly, memory (RAM) is also shared across all the cores. When we wish to implement an atomic instruction, we need to make sure the modified value is accessible to other cores. This, in general, requires modification in RAM. But since L2 (as well as L3) is shared, the hardware can decide to update it in L2 (or L3) and skip updating it in RAM. Accessing L2 is faster, hence the cost of atomic instruction (time to execute it) can significantly reduce.


### In L1 shared memory properties,its fixed 64kb for an sm does it mean L1 is divided equally among the threads available equally, what does configurable mean…?
> By default, L1 cache in CPUs is core-specific. Hence, it is not shared. Also, it is not accessible to the programmer -- one cannot say that I want a specific variable to be put into L1 cache.
> On GPUs, L1 cache is thread-block-specific. Hence, it is shared by all the block threads. How hardware manages it is unknown, but we can assume it to be equally divided across various threads in the block.
> However, on GPUs, L1 can be configured such that part of it is managed by the hardware, and part of it is available to the programmer. Thus, out of 64KB cache, I can configure it to 16KB for hardware and 48KB for programmers (or vice versa). We can assume that the hardware-managed part is divided equally among threads. However, the programmer-managed part is left to the programmer. It is like an array of 48KB (or 12K words) available for your thread block. All the thread-block threads can access the array as the programmer wants.


### In Bandwidth properties,” Big (wide) data bus rather than fast data bus ● Parallel data transfer ● Techniques to improve bandwidth: – Share / reuse data – Data compression – Recompute than store + fetch” what do they mean please explain briefly..
> Bus connects one piece of hardware to another. In our case, PCI-express bus connects CPU and GPU. All the communication between the two devices happens via this bus. One way to optimize this bus is to improve its speed of data transfer. Another way is to have a slower bus, but increase the number of parallel lanes. More the number of lanes, larger is the data transfer throughput. Another way is to increase the width of a single lane. Instead of having it deliver 8 bits at a time, the lane can be 32-bit wide. This also improves the throughput.
> To understand the other points, focus on a single lane. It has two wires, one for the incoming data and the other for outgoing. Parallel data transfer allows both these wires to be simultaneously used. Thus, we can copy data into the GPU via one wire, and out of GPU via another -- leading to improved throughput.
> Other ways to improve throughput is to avoid transferring data across devices. Thus, if some data is available on a device, we could reuse it for the next computation. Another way is to compress the data before transferring across the PCI-express bus. Since the amount of data would reduce due to compression, the bandwidth would increase.


### What is register spill ….?
> Consider a hardware with 10 registers. If I want to run a program on this hardware and I have 10 or fewer variables, I can store all these variables in the registers. This is useful because registers are on-chip and the program runs faster. However, if the program has 11 variables, I may have to store one variable in memory (RAM). RAM has a slower access compared to accessing a register. 
> The way it happens is the first ten variables which the program uses are stored in registers. When the 11th variable is required in the program, the hardware frees one register and brings the variable’s value into that freed register. Freeing a register involves storing the currently held value in the register to a location in memory. This is called a register spill -- the number of variables is large enough that the values spill from register to memory. Typically, the register which was used least recently is freed.


### What is throughput …?
> Throughput is a measure of performance: it is the amount of work done per unit time. One way to improve performance is to optimize per work-item processing (CPU style). Another way (GPU style) is to have slower work-item processing, but have many work-items processed by a larger number of threads.


### How Latency hiding on GPUs is done by exploiting massive multi-threading…?
> In CPUs, consider an 8-core machine running 8 threads. If a core is waiting for data to be made available due to a load instruction (LD X, R1 which loads the data stored in address of variable X from main memory into register R1), the core has to sit idle. Thus, a memory access incurs a high memory latency. On GPUs, when a thread stalls for data, there are lots of other threads waiting to be executed. Thus, the core can leave execution of the stalled thread, and start executing a ready thread. This improves core utilization and saves execution time. We say that we did not observe memory latency, as the core was always busy doing useful work. In other words, this hides latency. Later, when the data is available, the stalled thread becomes ready and can be taken up for execution. This requires that enough threads are available to be run on the GPU. Hence massive multi-threading helps here.


### what is warp stalling…?
> When a warp -- or a thread within a warp is waiting for a memory instruction to be completed (load or store), it gets stalled. This means that it is unable to proceed. In such a situation the multiprocessor takes up another ready warp for execution, and comes back to the stalled warp when it is ready.


### Not only reads but also write accesses are coalesced , please explain briefly.
> For the statement: `int x = a[threadIdx.x];` all the threads of the first warp read `a[0]..a[31]`. This is coalesced. For the statement, `a[threadIdx.x] = x;` all the warp-threads write to `a[0]..a[31]`. This is also coalesced. Thus, all the 32 writes are combined and completed in a single memory cycle / transaction.


### What is vectorization of intel processers how is it similar to memory coalescing….?
> In Intel CPUs, if you have the following instructions:
```a[i] = x; a[i+1] = x; a[i+2] = x; a[i+3] = x;```
> These can be executed one by one, or these could be combined into a single vector-instruction of width four. This improves the execution time.
> As you can notice, this is similar to coalescing by threads with warp-size = 4. A generalization of this with warp-size = 32 happens in GPUs.


### Intel compiler ICC automatically vectorizes good access patterns-what does it mean….?
> ICC is able to identify the above pattern a[i..i+k] for a k-word vector width. It automatically converts the original code (four different statements) into a single vector instruction. The programmer need not worry about it.


### If we pass a 2d array, say A[i][j] or  **A, during kernel launch,will it be automatically converted to 1d array while accessing the array in the kernel…?
> A[i][j] or **A would not be an array; it would be an element in this matrix.
> Yes, but if you pass A as a parameter, it can only be accessed as a single dimensional array in the kernel. 


### “In the GPU setting, another form of spacial locality is critical”, what is this another form of special locality…?
> It is coalescing. That means threads should access nearby elements.

### What is the difference between L1,shared memory, why does it matter to configure L1,shared memory, what is the application of this configuration option…?
> L1 is a hardware concept, its use in a CUDA program is referred to as shared memory. From a programmer’s perspective, they are the same. Consider each thread accessing `a[threadIdx.x]`. Hardware, while executing this instruction, would bring the data `a[threadIdx.x]` into L1 cache and access it. Since the access pattern is “easy”, it makes sense to leave it to the hardware to manage the cache. This is what happens in CPU, and also on GPU when we do not use shared memory.
> CUDA allows part of this L1 to be used as a scratchpad or programmable memory. On CPU, caches cannot be accessed from user-programs. On GPUs, using shared memory, L1 cache can be accessed. Thus, L1 cache can be split as (i) managed by hardware, as before, and (ii) managed by programmer, as shared memory. Out of 64KB available, 16+48 or 32+32 or 48+16 can be configured for L1+Shared.
> Shared memory can be helpful when it is difficult for the hardware runtime to identify the access pattern or need of various threads. For instance, if the access pattern is irregular (input-dependent, as in graph algorithms) or we use a memory pool for donation of work-items, etc., it is practically impossible for hardware to use cache effectively. Thus, hardware may end up caching an item which is not used later, or may decide to not cache an item because it doesn’t know that it would be used frequently later. In such situations, the programmer knows better about the access pattern, and can decide what elements to cache and what elements to not cache. This is where shared memory as scratchpad (like our usual array) be useful.


### Why threads in a warp accessing same word, is an exception for bank conflict…? please explain this point briefly.
> When threads access words from the same bank, there is a conflict. But when this word is the same accessed by a warp, then there is no conflict. Hence, this is an exception, despite the accesses being to the same bank.


### What are the applications for texture memory, what is its speciality…?
> It gets widely used in graphics applications wherein some small amount of 2D pixel data can be stored. Even in other applications, texture memory can be used to store small input data or configuration parameters, which do not change during kernel execution.
> The speciality of texture memory is that it is read-only, hence gets cached, and is faster than global memory (which is read-write). Also, it is designed in such a way that nearby 2D accesses (e.g., mat[i][j] and mat[i+1][j]) are faster.


### What is the speciality of constant memory, where,why is it used,is it like global memory where threads from all the blocks has access, threads on what level can access it(like all threads in a single block)or (threads in all the blocksa) etc..?
> Constant memory is also read-only, gets cached and hence faster to access. Thus, it can be used for storing small amount of data or configuration parameters, etc. It is used to improve performance. Yes, it is accessible from all the GPU threads -- similar to global memory.


### In compute capability concept of a GPU, what is” Used by the application at runtime (-arch=sm_35)”, “Macro __CUDA_ARCH__ is defined (e.g., 350) in device code”…?
> Compiler (nvcc) is capable of generating codes for different compute capabilities. Thus, a new nvcc can generate code for an older GPU. This is specified using -arch=sm_XY to nvcc on the command-line. The generated code can be run on a GPU supporting that compute capability.
> To make programs portable across various GPUs, CUDA allows programmers to use __CUDA_ARCH__ in the code. Thus, I can write the following:
```c
#if __CUDA_ARCH__ < 200
     // code for older GPUs
#elif __CUDA_ARCH__ < 500
    // middle-aged GPUs such as Kepler
#else
    // newer GPUs such as Volta
#endif
```
> This way, the rest of the code would remain the same, and the GPU-version-dependent code can be written using #if, and the whole code can co-exist. Another option is to write separate code for each GPU generation; but maintaining it would be difficult.


### What are “ Atomics int, float, Atomics double, warp-vote, __syncthreads functions, Unified memory, Dynamic parallelism”, should we know in detail about them or should we just acknowledge them and continue with the course…?
> You should know __syncthreads(). If you use fine-grained synchronization across threads, you should know about atomics. Others are a little advanced functions, which you should know for the course, but their use in your project would depend upon how much parallelism you wish to extract out of it.
> There is nothing like atomics int. Primitive types such as int, float, and (in newer GPUs) double get accessed atomically when we write to them using store instruction. I will leave their definitions to be seen from the slides.


### What is false sharing…?
> This is a concept from multi-core CPUs. Consider an instruction a[tid] = ... in a CPU thread. We know that all threads access different memory locations in this case. That is, there is no sharing of memory, hence there should not be any conflict. However, due to caching, when any thread i writes to a[i], the whole block containing a[i] is brought to the L1 cache of that core running thread i. Thus, if i = 10, a[0]..a[31] may be in the cache-line. Since one of the words in the cache line gets written to, the whole cache-line gets invalidated, and needs to be written back to global memory. The same scenario repeats on other cores for different threads. Thus, the time required to access independent memory locations is much higher -- due to this sequential access due to caching. This happens because although memory locations are distinct, their block is the same in the cache. Hence, it is called a false sharing. 
> For best performance, false sharing should be avoided by changing the access pattern to a[K*tid] where K makes sure consecutive threads write to different cache-lines.


### What exactly is an atomic section how is it different from critical section….?
> It is the same in our setup (although according to parallel programming concepts atomicity is different from mutual exclusion). However, in our setup, atomic instructions are different from critical sections implemented using locks. Atomic instructions are used to implement locks, using which we implement critical sections. In other words, atomic instructions are small and simple critical sections.


### __device__ volatile int n, volatile __shared__ int n; what does these statements mean…?
> ```__device__``` indicates a variable defined in the global memory of the GPU or a function executed on the GPU.
> ```__shared__``` indicates a variable stored in shared memory of the thread-block.
> volatile makes sure that the corresponding variable is always read from and written to global memory (or last level cache) on every access. Without volatile, the variable may be cached and may not be written to global memory -- as part of memory access optimization. volatile is required in thread synchronization, such as implementation of locks.


### In ways to remove datarace, aren’t   “1. Execute sequentially and 4. Mutual exclusion” mean the same thing…?
> Consider two threads, which both execute the code: S1; sum++; S2, where the initial value of sum is zero.
> If we execute threads T1 and T2 sequentially, S1 of T1 will never run in parallel with S1 of T2.
> f we execute them in parallel with mutual exclusion for sum++ (using locks or atomics), then S1 of the two threads can run in parallel, similarly, S1 of T1 may execute in parallel with S2 of T2 -- depending upon the speed of the two threads.
> In sequential execution, there is no need to use any locks or atomics, as there is never a data-race. This is how we write our usual C programs.


### What is graph library in GPU…?
> Graphs need to be implemented on GPUs, which can take care of creating an array based or pointer based representation. Similarly, various functions on graphs need to be implemented, such as, finding neighbors of a graph or updating attribute of a vertex or weight of an edge, etc. The memory representation of the graph along with these functions (API) constitute a graph library. It can be enhanced to include further graph API such as finding shortest paths or checking connectivity etc. 
> This is similar in spirit to having a string library (e.g., string.h) in C.


### “Atomics are primitive operations whose effects are visible either none or fully (never partially)” what does partially mean please explain with an example.
> Consider three statements: 
```c
load x, R1
Add R1, 1
Store R1, x
```
>This sequence of instructions increments the value stored in memory location x.
> Can you check the value of register R1 after Add instruction and before the Store? Yes.
> Such a flexibility is disallowed in atomic instructions or mutual exclusion blocks.
> For instance, atomicInc(&x, …) would disallow us from inspecting the value of x in-between -- while the instruction is still executing. We can either see the old value or the new value, but never anything in-between.
> Another longer example would make it clearer:
```c
x = 0;
lock();
x = 1;
x = 2;
x = 3;
unlock();
```
> In the above mutual-exclusion example, we will be able to see the value of x to be either 0 or 3, but never 1 or 2. Thus, none-effect is 0, full-effect is 3, partial effects are 1 and 2.


### What is a critrical section..?
> Critical section is a concept from operating systems wherein any shared memory access needs to be protected. Critical sections disallow dataraces in parallel programs. The lock-unlock example or atomic example are the critical sections.


### In one of the classworks “Implement single with atomicCAS”,what does single mean..?
> Single is an OpenMP construct from CPU domain wherein a piece of code, if executed by threads, is executed by only one thread out of all, and it does not matter which thread executes it.
> For instance, if the requirement is that threads perform some work, then one thread prints that work is done, this can be implemented using a single construct.
```c
#pragma omp parallel for
for (...) work();
single{ printf(“work done.\n”); }
```
> This makes sure that the printf is executed by exactly one thread.


### If there is no explicit global barrier then how can we make all the threads across blocks reach a point in the code or execute a line at a time….?
> The global barrier is not supported by CUDA, but needs to be implemented by programmers. I won’t give you the code, but the high-level idea.
> There are blockDim.x number of threads in each thread-block and gridDim.x number of thread blocks.
> Each block synchronizes with __syncthreads(). Now, one thread from each block (call it leader) synchronizes with leaders from other blocks using atomics. After this, each thread-block executes __syncthreads() again. This makes sure all the GPU threads have reached this point before executing the next instruction.


### what are locks, what are lock(), unlock() methods, how to implement them..?
> Say two threads T1 and T2 are executing the following program:
```c
1: t = x;
2: t = t + 1;
3: x = t;
```
> Here t is a variable local to each thread and x is shared. If the initial value of x is 0, we expect that after executing this program, the value of x should be incremented by 2. This happens when T1 fully executes the three statements, and then T2 executes, or vice versa.

> But let's say, the threads execute the statements in the following order:
```c
t1 = x;     // T1
t2 = x;     // T2
t1 = t1 + 1;     // T1
t2 = t2 + 1;     // T2
x = t1;     // T1
x = t2;     // T2
```
> Note that t is thread-local, so there are two copies of t (I have renamed those to t1 and t2 for clarity). But there is a single copy of x.
> So, both t1 and t2 are initialized to 0.
> Then both t1 and t2 are incremented to 1.
> Then T1 assigns 1 to x.
> Then T2 assigns 1 to x.

> Thus, at the end of the execution, the value of x is 1 (and not 2).
> This problem occurred because the three statements are not executed together -- but are interleaved with another thread.
> Thus, somehow we have to make sure that the three statements (in general, any number of statements) are executed together -- without any other statement interfering. Another way to say it is that the three statements are executed with mutual exclusion.

> This mutual exclusion is achieved in parallel programs using the notion of a lock.
> Thus, the program needs to be changed to:
```
lock();
t = x;
t = t + 1;
x = t;
unlock();
```
> This makes sure that the program gets executed ONLY as
> (i) three statements fully executed by T1, then by T2, or
> (ii) three statements fully executed by T2, then by T1.

> The question is how to implement the lock / unlock functions. This is done using peterson's algorithm or bakery algorithm.
> Peterson's algorithm works only for two threads, while Bakery algorithm works with arbitrary number of threads.


### What is preemption,what is a deadlock and how can preemption prevent it…?
> Preemption is the act of taking away some resource from a process even if that process has not finished its use. A deadlock occurs when a process is indefinitely waiting for other processes to release some resource (e.g, a lock guarding printer or a shared memory location), and the other processes are waiting for this process to release some other resource. The shortest example is with two processes P1 and P2. They both are interested in two resources R1 and R2. Say, P1 holds R1 (meaning, P1 locks R1) and P2 holds R2. Now, P1 wants to acquire R2, and P2 wants to acquire R1. 
> ```P1: lock(R1); lock(R2); useR1R2(); unlock(R2); unlock(R1);```
> ```P2: lock(R2); lock(R1); useR1R2(); unlock(R1); unlock(R2);```
> In this scenario, both P1 and P2 indefinitely wait for each other, and lead to deadlock. Deadlock situation means that no process is doing any useful work.
> Preemption allows the operating system to take away a resource from a process and make is available to others. For instance, preemption can release R1 from P1 and then P2 can get it. P2 can then execute, unlock R1 and R2, and then eventually, P1 can get both the resources. This would avoid the deadlock.
