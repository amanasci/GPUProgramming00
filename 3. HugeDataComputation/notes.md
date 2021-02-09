# Important Points:

## Order:
    1. generalconfig . cu 



## generalconfig . cu
1.  We take a simple approach for huge data. We will use 1D config for this. 
2.  Since thread limit in each block is 1024, we fix the `BLOCKSIZE` at 1024.
3.  Number of blocks to be launched will be calculated based on data size. 
4.  Then kernel will check for the number of data points and use only required number of threads to write.
5.   