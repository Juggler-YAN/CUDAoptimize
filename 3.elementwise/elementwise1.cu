#include <iostream>

#define THREAD_PER_BLOCK 256

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

__global__ void vec2_add(float* a, float* b, float* c) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    float2 reg_a = FETCH_FLOAT2(a[idx]);
    float2 reg_b = FETCH_FLOAT2(b[idx]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FETCH_FLOAT2(c[idx]) = reg_c;
}

int main() {

	// 1.����Host�ڴ�
	const int N = 1024 * 1024 * 32;
	const int BLOCK_PER_GRID = N / THREAD_PER_BLOCK;
	float* A_h = NULL;
	float* B_h = NULL;
	float* C_h = NULL;
	cudaMallocHost((void**)&A_h, N * sizeof(float));
	cudaMallocHost((void**)&B_h, N * sizeof(float));
	cudaMallocHost((void**)&C_h, N * sizeof(float));
	// 2.����Device�ڴ�
	float* A_d = NULL;
	float* B_d = NULL;
	float* C_d = NULL;
	cudaMalloc((void**)&A_d, N * sizeof(float));
	cudaMalloc((void**)&B_d, N * sizeof(float));
	cudaMalloc((void**)&C_d, N * sizeof(float));
	// 3.��ʼ��
	for (int i = 0; i < N; ++i) {
		A_h[i] = 1;
		B_h[i] = 1;
	}
	// 4.��Host�����ݿ�����Device��
	cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);
	// 5.kernel�˺���
	dim3 Grid(BLOCK_PER_GRID/2, 1);
	dim3 Block(THREAD_PER_BLOCK, 1);
	for (int i = 0; i < 10; ++i) {
		vec2_add << <Grid, Block >> > (A_d, B_d, C_d);
	}
	// 6.��Device�����ݿ�����Host��
	cudaMemcpy(C_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	// 7.����
	for (int i = 0; i < N; ++i) {
		if (C_h[i] != 2) {
			std::cout << "Wrong Result!!!" << std::endl;
			break;
		}
	}
	// 8.�ͷ��ڴ�
	cudaFreeHost(A_h);
	cudaFreeHost(B_h);
	cudaFreeHost(C_h);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
    
	return 0;
}