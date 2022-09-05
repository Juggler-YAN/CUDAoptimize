#include <iostream>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_X,
    const int THREAD_SIZE_Y,
    const bool ENABLE_DOUBLE_BUFFER
    >
__global__ void gemm(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by) * K];
    B = &B[BLOCK_SIZE_N * bx];

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i,
            A_TILE_COL,
            K
        )]);
        As[0][A_TILE_COL][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+3];
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i,
            B_TILE_COL,
            N
        )]);
    }
    __syncthreads();

    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }

    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K;
        if (tile_idx < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i,
                    A_TILE_COL + tile_idx,
                    K
                )]);
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i,
                    B_TILE_COL,
                    N
                )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K-1; ++j) {
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }
        
        if (tile_idx < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+3];
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y*ty+thread_y]);
        }
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X*tx+thread_x]);
        }
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++ thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N
            )]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

int main(int argc, char* argv[]) {

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    if (argc != 4) {
        std::cout << "usage: ./main [M] [K] [N]" << std::endl;
    }

    // 1.申请Host内存
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;
    float* A_h = NULL;
    float* B_h = NULL;
    float* C_h = NULL;
    cudaMallocHost((void**)&A_h, size_A * sizeof(float));
    cudaMallocHost((void**)&B_h, size_B * sizeof(float));
    cudaMallocHost((void**)&C_h, size_C * sizeof(float));
    // 2.申请device内存
    float* A_d = NULL;
    float* B_d = NULL;
    float* C_d = NULL;
    cudaMalloc((void**)&A_d, size_A * sizeof(float));
    cudaMalloc((void**)&B_d, size_B * sizeof(float));
    cudaMalloc((void**)&C_d, size_C * sizeof(float));
    // 3.初始化
    for (int i = 0; i < size_A; ++i) {
        A_h[i] = i / 13;
    }
    for (int i = 0; i < size_B; ++i) {
        B_h[i] = i % 13;
    }
    // 4.将Host中数据拷贝到Device中
    cudaMemcpy(A_d, A_h, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size_C * sizeof(float), cudaMemcpyHostToDevice);
	// 5.kernel核函数
    int n = 2000;
    for (int i = 0; i < n; ++i) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        gemm < BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_X, THREAD_SIZE_Y, ENABLE_DOUBLE_BUFFER >
        <<<dimGrid, dimBlock>>> (A_d, B_d, C_d, M, N, K);
    }
	// 6.将Device中数据拷贝到Host中
	cudaMemcpy(C_h, C_d, size_C * sizeof(float), cudaMemcpyDeviceToHost);
	// 7.释放内存
	cudaFreeHost(A_h);
	cudaFreeHost(B_h);
	cudaFreeHost(C_h);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
    
	return 0;
}