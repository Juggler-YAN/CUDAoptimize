#include <memory>
#include <iostream>
#include <cuda_runtime.h>

int main()
{
	int device_Count = 0;
	int device = 0;
	int driver_Version=0;
	cudaGetDeviceCount(&device_Count);   // 一个函数返回支持CUDA 的数量。没有返回0。
	cudaDeviceProp device_Property;
	cudaGetDeviceProperties(&device_Property, device);
	cudaDriverGetVersion(&driver_Version);
	if (device_Count == 0)
	{
		printf("没有支持CUDA 的设备\n");
	}
	else
	{
		printf("有 %d 个设备支持CUDA \n", device_Count);
	}
    
	printf("\n Device %d: \" %s \" \n ", device, device_Property.name);   // 设备型号

	printf(" CUDA Driver Version %d.%d\n", driver_Version / 1000, (driver_Version % 100) / 10);  // CUDA 版本
	// 显存大小
	printf(" Total amount of global memory : %.0f MB ( %llu bytes) \n", (float)device_Property.totalGlobalMem / 1048576.0f, (unsigned long long) device_Property.totalGlobalMem);

	printf(" (%2d) Multiprocessors\n", device_Property.multiProcessorCount);  // 最多含有多少多流处理器

	printf(" GPU Max Clock rate :`在这里插入代码片` %.0f MHz (%0.2f GHz ) \n", device_Property.clockRate * 1e-3f, device_Property.clockRate * 1e-6f);

	printf("Total amount of global memory : %.0f MB (%llu bytes) \n", (float)device_Property.totalGlobalMem / 1048576.0f, (unsigned long long ) device_Property.totalGlobalMem);

	printf("Memory Clock rate :%.0f Mhz \n ", device_Property.memoryClockRate * 1e-3f);
	printf("Memory Bus Width : %d-bit \n", device_Property.memoryBusWidth);   // 显存频率和显存位宽

	if (device_Property.l2CacheSize)
	{
		printf(" L2 Cache Size : %d bytes \n", device_Property.l2CacheSize);
	}
	printf("Total amout of constant memory : %lu bytes \n", device_Property.totalConstMem);   // 总常量显存

	printf("Total amount of shared memory per block : %lu bytes \n", device_Property.sharedMemPerBlock);
	printf("Total number of registers available per block : %d\n", device_Property.regsPerBlock);  //每个块的可用寄存器总数
	
	printf("Maximum number of threads per multiprocessor : %d\n ", device_Property.maxThreadsPerMultiProcessor);
	printf("Maximum number of threads per block : %d\n ", device_Property.maxThreadsPerBlock);

	printf("Max dimension size of a thread block (x,y,z) : (%d, %d, %d) \n ", 
					device_Property.maxThreadsDim[0], 
					device_Property.maxThreadsDim[1], 
					device_Property.maxThreadsDim[2]);

	printf("Max dimension size of a grid size (x,y,z) : ( %d, %d, %d) \n", 
					device_Property.maxGridSize[0], 
					device_Property.maxGridSize[1], 
					device_Property.maxGridSize[2]);

	printf(" ID or device : %d \n", device);

	memset(&device_Property, 0, sizeof(cudaDeviceProp));
	device_Property.major = 1;
	device_Property.minor = 3;
	cudaChooseDevice(&device, &device_Property);
	printf("ID of device which supports double precision id : %d \n", device);
	cudaSetDevice(device);

    return 0;
}
