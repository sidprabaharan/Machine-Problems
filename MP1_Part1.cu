/*
 * ELEC 374 - Digital Systems Engineering
 * Machine Problem 1 - Part 1: Device Query
 * Student Name: Sid Prabaharan
 * Student ID: 20351244
 *
 */

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Get number of CUDA devices
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf(" Failed to get device count - %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Check if any CUDA devices were found
    if (deviceCount == 0) {
        printf("No CUDA capable devices found\n");
        return 1;
    }

    printf("Found %d CUDA devices\n\n", deviceCount);

    // Iterate through all detected devices
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("Device %d: \"%s\"\n", i, deviceProp.name);
        printf("--------------------------------------------------\n");

        // Get the device clock rate in MHz
        printf("Clock rate: %.2f MHz\n", deviceProp.clockRate * 1e-3f);

        // Number of streaming multiprocessors (SMs)
        printf("Number of streaming multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);

        // Calculate cores based on compute capability and SM count
        int coresPerSM = 0;
        if (deviceProp.major == 3) {      
            coresPerSM = 192;
        }
        else if (deviceProp.major == 5) { 
            coresPerSM = 128;
        }
        else if (deviceProp.major == 6) { 
            coresPerSM = 64;
            if (deviceProp.minor == 1) coresPerSM = 128;
        }
        else if (deviceProp.major == 7) { 
            if (deviceProp.minor == 0) coresPerSM = 64;
            else coresPerSM = 64;
        }
        else if (deviceProp.major == 8) { 
            coresPerSM = 64;
            if (deviceProp.minor == 6) coresPerSM = 128; 
        }
        else if (deviceProp.major == 9) { 
            coresPerSM = 128;
        }
        else {
            coresPerSM = 32; 
        }

        int totalCores = coresPerSM * deviceProp.multiProcessorCount;
        printf("Number of CUDA cores: %d\n", totalCores);

        // Warp size
        printf("Warp size: %d\n", deviceProp.warpSize);

        // Memory information
        printf("Global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Constant memory: %.2f KB\n", deviceProp.totalConstMem / 1024.0);
        printf("Shared memory per block: %.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0);

        // Registers
        printf("Number of registers available per block: %d\n", deviceProp.regsPerBlock);

        // Thread information
        printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);

        // Maximum dimensions
        printf("Maximum size of each dimension of a block: [%d, %d, %d]\n",
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);

        printf("Maximum size of each dimension of a grid: [%d, %d, %d]\n",
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);

        printf("\n");
    }

    return 0;
}
