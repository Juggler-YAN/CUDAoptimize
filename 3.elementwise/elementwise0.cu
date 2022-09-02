#include <iostream>

#define THREAD_PER_BLOCK 256

__global__ void add(float* a, float* b, float* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}

int main() {

	// 1.申请Host内存
	const int N = 1024 * 1024 * 32;
	const int BLOCK_PER_GRID = N / THREAD_PER_BLOCK;
	float* A_h = NULL;
	float* B_h = NULL;
	float* C_h = NULL;
	cudaMallocHost((void**)&A_h, N * sizeof(float));
	cudaMallocHost((void**)&B_h, N * sizeof(float));
	cudaMallocHost((void**)&C_h, N * sizeof(float));
	// 2.申请Device内存
	float* A_d = NULL;
	float* B_d = NULL;
	float* C_d = NULL;
	cudaMalloc((void**)&A_d, N * sizeof(float));
	cudaMalloc((void**)&B_d, N * sizeof(float));
	cudaMalloc((void**)&C_d, N * sizeof(float));
	// 3.初始化
	for (int i = 0; i < N; ++i) {
		A_h[i] = 1;
		B_h[i] = 1;
	}
	// 4.将Host中数据拷贝到Device中
	cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);
	// 5.kernel核函数
	dim3 Grid(BLOCK_PER_GRID, 1);
	dim3 Block(THREAD_PER_BLOCK, 1);
	for (int i = 0; i < 10; ++i) {
		add << <Grid, Block >> > (A_d, B_d, C_d);
	}
	// 6.将Device中数据拷贝到Host中
	cudaMemcpy(C_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	// 7.后处理
	for (int i = 0; i < N; ++i) {
		if (C_h[i] != 2) {
			std::cout << "Wrong Result!!!" << std::endl;
			break;
		}
	}
	// 8.释放内存
	cudaFreeHost(A_h);
	cudaFreeHost(B_h);
	cudaFreeHost(C_h);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
    
	return 0;
}