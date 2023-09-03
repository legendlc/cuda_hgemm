#include <cstdio>
#include <cuda.h>

int main()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Clock Rate (MHz): %d\n", prop.clockRate/1024);
        printf("  With %d multiprocessors.\n", prop.multiProcessorCount);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Memory Clock Rate (MHz): %d\n",
               prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  L2 cache size: %d KB\n", prop.l2CacheSize/1024);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max size of each dimension of a block: %d %d %d\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max size of each dimension of a grid: %d %d %d\n",
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }

    return 0;
}