#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) {
    int decive_id = 0;
    if (argc > 1) {
        decive_id = atoi(argv[1]);
    }
    // 设置主机当前所要用的设备
    cudaSetDevice(decive_id);
    cudaDeviceProp prop;
    // 获取设备信息
    cudaGetDeviceProperties(&prop, decive_id);
    printf("Device id: %d\n", decive_id);
    printf("设备名称: %s\n", prop.name);
    printf("每个线程块的最大寄存器数: %d\n", prop.regsPerBlock);
    printf("每个流处理器中的最大寄存器数: %d\n",prop.regsPerMultiprocessor);
    printf("一个线程束中包含的最大线程数量: %d\n", prop.warpSize);
    printf("一个线程块中可以包含的最大线程数量: %d\n", prop.maxThreadsPerBlock);
    printf("在多维线程块数组中，每一维可以包含的最大线程数量: %d, %d, %d\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("在一个线程格中，每一维可以包含的线程块数量: %d, %d, %d\n", 
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("设备上流多处理器的数量(SM): %d\n", prop.multiProcessorCount);
    printf("设备上全局内存总量: %g GB\n", prop.totalGlobalMem/(1024.0*1024*1024));
    printf("在一个线程块中可使用的最大共享内存数量数量: %g KB\n",
        prop.sharedMemPerBlock/1024.0);
    printf("每个SM的最大共享内存: %g KB\n", prop.sharedMemPerMultiprocessor/1024.);
    printf("常量内存总量: %g KB\n", prop.totalConstMem/1024.0);

    return 0;
}