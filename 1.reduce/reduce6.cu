#include <iostream>

#define THREAD_PER_BLOCK 256

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid+32];
	if (blockSize >= 32) sdata[tid] += sdata[tid+16];
	if (blockSize >= 16) sdata[tid] += sdata[tid+8];
	if (blockSize >= 8) sdata[tid] += sdata[tid+4];
	if (blockSize >= 4) sdata[tid] += sdata[tid+2];
	if (blockSize >= 2) sdata[tid] += sdata[tid+1];
}

template <unsigned int blockSize, int NUM_THREAD>
__global__ void reduce6(float* g_idata, float* g_odata, unsigned int n) {

	__shared__ float sdata[blockSize];

	// ÿһ���̴߳�ȫ���ڴ�װ��һ��Ԫ�ص������ڴ�
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize*NUM_THREAD) + threadIdx.x;

	sdata[tid] = 0;

	#pragma unroll
	for (int iter = 0; iter < NUM_THREAD; ++iter) {
		sdata[tid] += g_idata[i+iter*blockSize];
	}

	__syncthreads();

	// �ڹ����ڴ���ִ��reduce����
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce<blockSize>(sdata, tid);

	// ���ÿ�Ľ��д��ȫ���ڴ�
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

int main() {

	// 1.����Host�ڴ�
	const int N = 1024 * 1024 * 32;
	const int BLOCK_PER_GRID = 1024;
	const int NUM_BLOCK = N / BLOCK_PER_GRID;
	const int NUM_THREAD = NUM_BLOCK / THREAD_PER_BLOCK;
	float* A_h = NULL;
	float* Aout_h = NULL;
	cudaMallocHost((void**)&A_h, N * sizeof(float));
	cudaMallocHost((void**)&Aout_h, BLOCK_PER_GRID * sizeof(float));
	// 2.����Device�ڴ�
	float* A_d = NULL;
	float* Aout_d = NULL;
	cudaMalloc((void**)&A_d, N * sizeof(float));
	cudaMalloc((void**)&Aout_d, BLOCK_PER_GRID * sizeof(float));
	// 3.��ʼ��
	for (int i = 0; i < N; ++i) {
		A_h[i] = 1;
	}
	// 4.��Host�����ݿ�����Device��
	cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
	// 5.kernel�˺���
	dim3 Grid(BLOCK_PER_GRID, 1);
	dim3 Block(THREAD_PER_BLOCK, 1);
	reduce6<THREAD_PER_BLOCK, NUM_THREAD> << <Grid, Block >> > (A_d, Aout_d, N);
	// 6.��Device�����ݿ�����Host��
	cudaMemcpy(Aout_h, Aout_d, BLOCK_PER_GRID * sizeof(float), cudaMemcpyDeviceToHost);
	// 7.����
	for (int i = 0; i < BLOCK_PER_GRID; ++i) {
		if (Aout_h[i] != NUM_BLOCK) {
			std::cout << "Wrong Result!!!" << std::endl;
			break;
		}
	}
	// 8.�ͷ��ڴ�
	cudaFreeHost(A_h);
	cudaFreeHost(Aout_h);
	cudaFree(A_d);
	cudaFree(Aout_d);
    
	return 0;
}