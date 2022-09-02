#include <iostream>

#define THREAD_PER_BLOCK 256

__global__ void reduce2(float* g_idata, float* g_odata) {

	__shared__ float sdata[THREAD_PER_BLOCK];

	// ÿһ���̴߳�ȫ���ڴ�װ��һ��Ԫ�ص������ڴ�
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// �ڹ����ڴ���ִ��reduce����
	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// ���ÿ�Ľ��д��ȫ���ڴ�
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

int main() {

	// 1.����Host�ڴ�
	const int N = 1024 * 1024 * 32;
	const int BLOCK_PER_GRID = ceil(static_cast<float>(N) / THREAD_PER_BLOCK);
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
	reduce2 << <Grid, Block >> > (A_d, Aout_d);
	// 6.��Device�����ݿ�����Host��
	cudaMemcpy(Aout_h, Aout_d, BLOCK_PER_GRID * sizeof(float), cudaMemcpyDeviceToHost);
	// 7.����
	for (int i = 0; i < BLOCK_PER_GRID; ++i) {
		if (Aout_h[i] != THREAD_PER_BLOCK) {
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