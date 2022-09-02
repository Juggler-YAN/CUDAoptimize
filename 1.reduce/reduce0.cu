#include <iostream>

#define THREAD_PER_BLOCK 256

__global__ void reduce0(float* g_idata, float* g_odata) {

	__shared__ float sdata[THREAD_PER_BLOCK];

	// 每一个线程从全局内存装载一个元素到共享内存
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// 在共享内存上执行reduce计算
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// 将该块的结果写到全局内存
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

int main() {

	// 1.申请Host内存
	const int N = 1024 * 1024 * 32;
	const int BLOCK_PER_GRID = ceil(static_cast<float>(N) / THREAD_PER_BLOCK);
	float* A_h = NULL;
	float* Aout_h = NULL;
	cudaMallocHost((void**)&A_h, N * sizeof(float));
	cudaMallocHost((void**)&Aout_h, BLOCK_PER_GRID * sizeof(float));
	// 2.申请Device内存
	float* A_d = NULL;
	float* Aout_d = NULL;
	cudaMalloc((void**)&A_d, N * sizeof(float));
	cudaMalloc((void**)&Aout_d, BLOCK_PER_GRID * sizeof(float));
	// 3.初始化
	for (int i = 0; i < N; ++i) {
		A_h[i] = 1;
	}
	// 4.将Host中数据拷贝到Device中
	cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
	// 5.kernel核函数
	dim3 Grid(BLOCK_PER_GRID, 1);
	dim3 Block(THREAD_PER_BLOCK, 1);
	reduce0 << <Grid, Block >> > (A_d, Aout_d);
	// 6.将Device中数据拷贝到Host中
	cudaMemcpy(Aout_h, Aout_d, BLOCK_PER_GRID * sizeof(float), cudaMemcpyDeviceToHost);
	// 7.后处理
	for (int i = 0; i < BLOCK_PER_GRID; ++i) {
		if (Aout_h[i] != THREAD_PER_BLOCK) {
			std::cout << "Wrong Result!!!" << std::endl;
			break;
		}
	}
	// 8.释放内存
	cudaFreeHost(A_h);
	cudaFreeHost(Aout_h);
	cudaFree(A_d);
	cudaFree(Aout_d);
    
	return 0;
}