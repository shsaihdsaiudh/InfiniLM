// fp8_blockwise GEMM kernel 调优试验台（独立于 InfiniCore 构建,快速迭代)
// 变体:stream(纯流上限)/ v2(当前实现)/ v3(行交错双行)/ v4(warp单行,TN=8)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                            \
    do {                                                                 \
        cudaError_t e_ = (x);                                            \
        if (e_ != cudaSuccess) {                                         \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_),   \
                   __FILE__, __LINE__);                                  \
            exit(1);                                                     \
        }                                                                \
    } while (0)

__device__ __forceinline__ float fp8_decode(uint8_t value) {
    const uint32_t magnitude = value & 0x7fU;
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & 0x7U;
    if (exponent == 0xfU && mantissa == 0x7U) {
        return __uint_as_float(0x7fffffffU);
    }
    const float decoded = exponent == 0U
                            ? static_cast<float>(mantissa) * 0.001953125f
                            : __uint_as_float((exponent + 120U) << 23U) * (1.0f + static_cast<float>(mantissa) * 0.125f);
    return (value & 0x80U) ? -decoded : decoded;
}

__device__ __forceinline__ void decode16(uint4 q16, float *w) {
    const uint32_t u[4] = {q16.x, q16.y, q16.z, q16.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i * 4 + 0] = fp8_decode(u[i] & 0xffU);
        w[i * 4 + 1] = fp8_decode((u[i] >> 8) & 0xffU);
        w[i * 4 + 2] = fp8_decode((u[i] >> 16) & 0xffU);
        w[i * 4 + 3] = fp8_decode(u[i] >> 24);
    }
}

__device__ __forceinline__ void load16_bf16(const __nv_bfloat16 *p, float *v) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 b01 = *reinterpret_cast<const __nv_bfloat162 *>(p + i * 4);
        const __nv_bfloat162 b23 = *reinterpret_cast<const __nv_bfloat162 *>(p + i * 4 + 2);
        const float2 f01 = __bfloat1622float2(b01);
        const float2 f23 = __bfloat1622float2(b23);
        v[i * 4 + 0] = f01.x;
        v[i * 4 + 1] = f01.y;
        v[i * 4 + 2] = f23.x;
        v[i * 4 + 3] = f23.y;
    }
}

// ---------------------------------------------------------------------------
// A) 纯流上限:只读 Q,不做计算
// ---------------------------------------------------------------------------
template <int ROWS_PER_WARP>
__global__ void stream_kernel(const uint8_t *__restrict__ q, float4 *__restrict__ sink, size_t N, size_t K) {
    const size_t n0 = static_cast<size_t>(blockIdx.x) * ((blockDim.x >> 5) * ROWS_PER_WARP);
    if (n0 >= N) { return; }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t k_groups = K / 512;
    const size_t lane_off = static_cast<size_t>(lane) * 16;
    uint4 acc = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int rw = 0; rw < ROWS_PER_WARP; ++rw) {
        const uint8_t *q_row = q + (n0 + warp * ROWS_PER_WARP + rw) * K;
        for (size_t kg = 0; kg < k_groups; ++kg) {
            const uint4 v = *reinterpret_cast<const uint4 *>(q_row + kg * 512 + lane_off);
            acc.x ^= v.x;
            acc.y ^= v.y;
            acc.z ^= v.z;
            acc.w ^= v.w;
        }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xdeadbeefU) {
        sink[threadIdx.x] = make_float4(1.f, 2.f, 3.f, 4.f);
    }
}

// ---------------------------------------------------------------------------
// B) v2 当前实现(顺序两行)
// ---------------------------------------------------------------------------
__global__ void fused_v2(const __nv_bfloat16 *__restrict__ out, const __nv_bfloat16 *__restrict__ a,
                         const uint8_t *__restrict__ q, const float *__restrict__ scales,
                         size_t M, size_t N, size_t K, size_t block_n, size_t block_k, size_t scales_cols,
                         __nv_bfloat16 *__restrict__ out_) {
    const size_t n0 = static_cast<size_t>(blockIdx.x) * 16;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int rw = 0; rw < 2; ++rw) {
        const size_t row = n0 + warp * 2 + rw;
        const uint8_t *q_row = q + row * K;
        const size_t scale_row = row / block_n;
        float acc = 0.0f;
        const size_t k_groups = K / 512;
        const size_t lane_off = static_cast<size_t>(lane) * 16;
        const size_t sub_chunk = static_cast<size_t>(lane) / 8;
        uint4 q_next = *reinterpret_cast<const uint4 *>(q_row + lane_off);
        for (size_t kg = 0; kg < k_groups; ++kg) {
            const uint4 q_cur = q_next;
            if (kg + 1 < k_groups) {
                q_next = *reinterpret_cast<const uint4 *>(q_row + (kg + 1) * 512 + lane_off);
            }
            float w[16];
            decode16(q_cur, w);
            const float scale = scales[scale_row * scales_cols + (kg * 512 + sub_chunk * 128) / block_k];
            float av[16];
            load16_bf16(a + kg * 512 + lane_off, av);
            float cacc = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                cacc = fmaf(w[j], av[j], cacc);
            }
            acc = fmaf(scale, cacc, acc);
        }
        float v = acc;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
            out_[row] = __float2bfloat16_rn(v);
        }
    }
    (void)out;
}

// ---------------------------------------------------------------------------
// C) v3:两行交错(每行组同时发两个 uint4,提高 MLP)
// ---------------------------------------------------------------------------
__global__ void fused_v3(const __nv_bfloat16 *__restrict__ a,
                         const uint8_t *__restrict__ q, const float *__restrict__ scales,
                         size_t N, size_t K, size_t block_n, size_t block_k, size_t scales_cols,
                         __nv_bfloat16 *__restrict__ out_) {
    const size_t n0 = static_cast<size_t>(blockIdx.x) * 16;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row0 = n0 + warp * 2;
    const uint8_t *q_row0 = q + row0 * K;
    const uint8_t *q_row1 = q_row0 + K;
    const size_t k_groups = K / 512;
    const size_t lane_off = static_cast<size_t>(lane) * 16;
    const size_t sub_chunk = static_cast<size_t>(lane) / 8;

    float acc0 = 0.0f, acc1 = 0.0f;
    uint4 qa0 = *reinterpret_cast<const uint4 *>(q_row0 + lane_off);
    uint4 qa1 = *reinterpret_cast<const uint4 *>(q_row1 + lane_off);
    for (size_t kg = 0; kg < k_groups; ++kg) {
        const uint4 q0 = qa0, q1 = qa1;
        if (kg + 1 < k_groups) {
            qa0 = *reinterpret_cast<const uint4 *>(q_row0 + (kg + 1) * 512 + lane_off);
            qa1 = *reinterpret_cast<const uint4 *>(q_row1 + (kg + 1) * 512 + lane_off);
        }
        float w0[16], w1[16];
        decode16(q0, w0);
        decode16(q1, w1);
        float av[16];
        load16_bf16(a + kg * 512 + lane_off, av);
        float c0 = 0.0f, c1 = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            c0 = fmaf(w0[j], av[j], c0);
            c1 = fmaf(w1[j], av[j], c1);
        }
        const float s0 = scales[(row0 / block_n) * scales_cols + (kg * 512 + sub_chunk * 128) / block_k];
        const float s1 = scales[((row0 + 1) / block_n) * scales_cols + (kg * 512 + sub_chunk * 128) / block_k];
        acc0 = fmaf(s0, c0, acc0);
        acc1 = fmaf(s1, c1, acc1);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_xor_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0) {
        out_[row0] = __float2bfloat16_rn(acc0);
        out_[row0 + 1] = __float2bfloat16_rn(acc1);
    }
}

// ---------------------------------------------------------------------------
// D) v4:warp 单行,TN=8,128 线程,块数翻倍
// ---------------------------------------------------------------------------
__global__ void fused_v4(const __nv_bfloat16 *__restrict__ a,
                         const uint8_t *__restrict__ q, const float *__restrict__ scales,
                         size_t N, size_t K, size_t block_n, size_t block_k, size_t scales_cols,
                         __nv_bfloat16 *__restrict__ out_) {
    const size_t row = static_cast<size_t>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    if (row >= N) { return; }
    const int lane = threadIdx.x & 31;
    const uint8_t *q_row = q + row * K;
    const size_t k_groups = K / 512;
    const size_t lane_off = static_cast<size_t>(lane) * 16;
    const size_t sub_chunk = static_cast<size_t>(lane) / 8;
    float acc = 0.0f;
    uint4 q_next = *reinterpret_cast<const uint4 *>(q_row + lane_off);
    for (size_t kg = 0; kg < k_groups; ++kg) {
        const uint4 q_cur = q_next;
        if (kg + 1 < k_groups) {
            q_next = *reinterpret_cast<const uint4 *>(q_row + (kg + 1) * 512 + lane_off);
        }
        float w[16];
        decode16(q_cur, w);
        float av[16];
        load16_bf16(a + kg * 512 + lane_off, av);
        float cacc = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            cacc = fmaf(w[j], av[j], cacc);
        }
        const float scale = scales[(row / block_n) * scales_cols + (kg * 512 + sub_chunk * 128) / block_k];
        acc = fmaf(scale, cacc, acc);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        out_[row] = __float2bfloat16_rn(acc);
    }
}


// ---------------------------------------------------------------------------
// E) v5:v4 + 双组展开(每 warp 同时在途 2 组 uint4)
// ---------------------------------------------------------------------------
__global__ void fused_v5(const __nv_bfloat16 *__restrict__ a,
                         const uint8_t *__restrict__ q, const float *__restrict__ scales,
                         size_t N, size_t K, size_t block_n, size_t block_k, size_t scales_cols,
                         __nv_bfloat16 *__restrict__ out_) {
    const size_t row = static_cast<size_t>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    if (row >= N) { return; }
    const int lane = threadIdx.x & 31;
    const uint8_t *q_row = q + row * K;
    const size_t k_groups = K / 512;
    const size_t lane_off = static_cast<size_t>(lane) * 16;
    const size_t sub_chunk = static_cast<size_t>(lane) / 8;
    float acc = 0.0f;
    uint4 q0 = *reinterpret_cast<const uint4 *>(q_row + lane_off);
    uint4 q1 = *reinterpret_cast<const uint4 *>(q_row + 512 + lane_off);
    for (size_t kg = 0; kg < k_groups; kg += 2) {
        const uint4 qc0 = q0, qc1 = q1;
        if (kg + 2 < k_groups) {
            q0 = *reinterpret_cast<const uint4 *>(q_row + (kg + 2) * 512 + lane_off);
            q1 = *reinterpret_cast<const uint4 *>(q_row + (kg + 3) * 512 + lane_off);
        }
        float av0[16], av1[16];
        load16_bf16(a + kg * 512 + lane_off, av0);
        load16_bf16(a + (kg + 1) * 512 + lane_off, av1);
        float w0[16], w1[16];
        decode16(qc0, w0);
        decode16(qc1, w1);
        float c0 = 0.0f, c1 = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            c0 = fmaf(w0[j], av0[j], c0);
            c1 = fmaf(w1[j], av1[j], c1);
        }
        const size_t kb0 = kg * 512 + sub_chunk * 128;
        const size_t kb1 = (kg + 1) * 512 + sub_chunk * 128;
        const float s0 = scales[(row / block_n) * scales_cols + kb0 / block_k];
        const float s1 = scales[(row / block_n) * scales_cols + kb1 / block_k];
        acc = fmaf(s0, c0, acc);
        acc = fmaf(s1, c1, acc);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        out_[row] = __float2bfloat16_rn(acc);
    }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------

template <typename F>
static float bench(F fn, int iters, cudaStream_t stream) {
    for (int i = 0; i < 20; ++i) {
        fn(stream);
    }
    CK(cudaStreamSynchronize(stream));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) {
        fn(stream);
    }
    CK(cudaEventRecord(t1, stream));
    CK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms / iters;
}

static uint8_t *d_q;
static size_t g_slab = 0, g_slab_count = 24, g_slab_bytes = 0;
static __nv_bfloat16 *d_a, *d_out;
static float *d_scales, *d_sink4;
static size_t N, K;
static cudaStream_t g_stream;

int main() {
    N = 4096;
    K = 4096;
    const size_t M = 1, block_n = 128, block_k = 128;
    const size_t scales_cols = K / block_k;

    CK(cudaStreamCreate(&g_stream));
    const size_t Q_SLABS = 24; // 24 x 16.8MB = 403MB,远超 L2
    CK(cudaMalloc(&d_q, N * K * Q_SLABS));
    CK(cudaMalloc(&d_a, M * K * 2));
    CK(cudaMalloc(&d_out, N * 2 * 4));
    CK(cudaMalloc(&d_scales, (N / block_n) * scales_cols * sizeof(float)));
    CK(cudaMalloc(&d_sink4, 1024 * sizeof(float4)));

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> db(0, 255);
    std::vector<uint8_t> hq(N * K);
    for (auto &v : hq) {
        v = static_cast<uint8_t>(db(rng) & 0x7f); // 避免 NaN 位型的影响
    }
    for (size_t s = 0; s < Q_SLABS; ++s) {
        CK(cudaMemcpy(d_q + s * N * K, hq.data(), hq.size(), cudaMemcpyHostToDevice));
    }
    std::vector<uint16_t> ha(M * K);
    std::uniform_real_distribution<float> df(-0.5f, 0.5f);
    for (auto &v : ha) {
        const float f = df(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        v = static_cast<uint16_t>(u >> 16);
    }
    CK(cudaMemcpy(d_a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> hs((N / block_n) * scales_cols);
    for (auto &v : hs) {
        v = 0.005f + 0.01f * df(rng);
    }
    CK(cudaMemcpy(d_scales, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));

    const double bytes = static_cast<double>(N) * K;
    g_slab_bytes = N * K;
#define NEXT_Q (d_q + (g_slab = (g_slab + 1) % g_slab_count) * g_slab_bytes)
    char name[64];

    {
        snprintf(name, 64, "stream_2row");
        auto fn = [&](cudaStream_t s) { stream_kernel<2><<<N / 16, 256, 0, s>>>(NEXT_Q, reinterpret_cast<float4 *>(d_sink4), N, K); };
        const float us = bench(fn, 300, g_stream) * 1000.f;
        printf("%-14s: %8.2f us  %7.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    }
    {
        snprintf(name, 64, "fused_v2");
        auto fn = [&](cudaStream_t s) { fused_v2<<<N / 16, 256, 0, s>>>(d_out, d_a, NEXT_Q, d_scales, 1, N, K, block_n, block_k, scales_cols, d_out); };
        const float us = bench(fn, 300, g_stream) * 1000.f;
        printf("%-14s: %8.2f us  %7.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    }
    {
        snprintf(name, 64, "fused_v3_ilv");
        auto fn = [&](cudaStream_t s) { fused_v3<<<N / 16, 256, 0, s>>>(d_a, NEXT_Q, d_scales, N, K, block_n, block_k, scales_cols, d_out); };
        const float us = bench(fn, 300, g_stream) * 1000.f;
        printf("%-14s: %8.2f us  %7.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    }
    {
        snprintf(name, 64, "stream_tn8");
        auto fn = [&](cudaStream_t s) {
            const uint8_t *qp = NEXT_Q;
            // warp 单行、128 线程几何的纯流
            stream_kernel<1><<<N / 4, 128, 0, s>>>(qp, reinterpret_cast<float4 *>(d_sink4), N, K);
        };
        const float us = bench(fn, 300, g_stream) * 1000.f;
        printf("%-14s: %8.2f us  %7.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    }
    {
        snprintf(name, 64, "fused_v4_tn8");
        auto fn = [&](cudaStream_t s) { fused_v4<<<N / 4, 128, 0, s>>>(d_a, NEXT_Q, d_scales, N, K, block_n, block_k, scales_cols, d_out); };
        const float us = bench(fn, 300, g_stream) * 1000.f;
        printf("%-14s: %8.2f us  %7.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    }

    {
        snprintf(name, 64, "fused_v5_u2");
        auto fn = [&](cudaStream_t s) { fused_v5<<<N / 4, 128, 0, s>>>(d_a, NEXT_Q, d_scales, N, K, block_n, block_k, scales_cols, d_out); };
        const float us = bench(fn, 300, g_stream) * 1000.f;
        printf("%-14s: %8.2f us  %7.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    }

    CK(cudaDeviceSynchronize());
    printf("TUNE_DONE\n");
    return 0;
}
