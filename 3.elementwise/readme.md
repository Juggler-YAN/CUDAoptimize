# elementwise算子优化
逐位进行运算，例如求向量和<br>
优化方式：向量化访存

base的实现

``` CUDA
__global__ void add(float* a, float* b, float* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
```

优化：float2访问内存 <br>

``` CUDA
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
```

优化：float4访问内存 <br>

``` CUDA
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
__global__ void vec4_add(float* a, float* b, float* c) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(c[idx]) = reg_c;
}
```


|          | 带宽 |
| :-----:  | :-------: |
| elementwise0  | 92.15 |
| elementwise1  | 92.18 |
| elementwise2  | 92.65 |

总结：elmentwise类的kernel正确地设置好block和thread，使用向量化访存，应该就能到达理论极限性能。