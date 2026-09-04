// cuBLASLt block-scale FP8 (DeepSeek 式 [1x128] / [128x128] block scaling) 在
// sm_120 上的可行性 spike —— 独立于 InfiniCore 构建,服务器上直接编译运行:
//
//   nvcc -O2 -arch=native fp8_cublaslt_spike.cu -o fp8_cublaslt_spike -lcublasLt
//   ./fp8_cublaslt_spike
//
// 回答三个问题:
//   1) 服务器 CUDA/cuBLASLt 版本是否有 VEC128_32F / BLK128x128_32F API(需 >= 12.9);
//   2) sm_120 上 cublasLtMatmulAlgoGetHeuristic 是否返回可用 algo(该模式文档
//      标注 Hopper;Blackwell 消费卡是否放行要实测);
//   3) 若有解:数值是否正确(scale 布局是否符合文档)、速度相对自研 mma kernel
//      目标值(bs=8..32 decode GEMM)如何。
//
// 背景:cuBLASLt block scaling 要求两个乘数都是 FP8(不支持 BF16xFP8 混合),
// 所以该路线 = W8A8:激活需先在线量化到 e4m3(每行每 128 一个 scale),权重沿用
// 现有 [N/128, K/128] block scale。算子映射(行主序世界 -> cuBLASLt 列主序):
//   C_rm[M,N] = X_rm[M,K] * W_rm[N,K]^T
//   <=> cublasLt: m=N, n=M, k=K
//       A 操作数 = W(KxN col-major, ld=K, OP_T), scale 模式 BLK128x128_32F
//       B 操作数 = X(KxM col-major, ld=K, OP_N), scale 模式 VEC128_32F
//       C       = out(NxM col-major, ld=N)
//   文档布局要求(已在此落实):
//     BLK128x128: K-major, 形状 L4 x ceil(N/128), L4=ceil(L/4)*4
//                 -> 恰好是现有 [N/128, K/128] 行主序 padding 到 L4 列;
//     VEC128:     B 为 N-major(n=M 连续) -> 激活 scale 按 [L][M] 写;
//     M、N(cublasLt 维度)必须是 4 的倍数 -> decode batch 需 pad 到 4 的倍数。
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cmath>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK_CUDA(x)                                                           \
    do {                                                                     \
        cudaError_t e_ = (x);                                                \
        if (e_ != cudaSuccess) {                                             \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_),       \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define CK_LT(x)                                                             \
    do {                                                                     \
        cublasStatus_t s_ = (x);                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                   \
            printf("cuBLASLt error %d at %s:%d\n", (int)s_, __FILE__,        \
                   __LINE__);                                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// CPU 端 e4m3 编解码(spike 精度足够即可;数值校验以编码后的码值为准)
// ---------------------------------------------------------------------------

static float e4m3_decode(uint8_t v) {
    const int s = (v >> 7) & 1;
    const int e = (v >> 3) & 0xF;
    const int m = v & 7;
    if (e == 0xF && m == 7) {
        return NAN;
    }
    float val = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return s ? -val : val;
}

static uint8_t e4m3_encode(float f) {
    if (std::isnan(f)) {
        return 0x7F;
    }
    uint8_t s = f < 0 ? 0x80 : 0;
    float a = fabsf(f);
    if (a > 448.0f) {
        a = 448.0f;
    }
    if (a < ldexpf(1.0f, -9)) {
        return s; // 小于最小 denormal 一半 -> 0
    }
    if (a < ldexpf(1.0f, -6)) { // denormal
        int m = (int)lrintf(a / ldexpf(1.0f, -9));
        if (m > 7) {
            m = 7;
        }
        return s | (uint8_t)m;
    }
    int e = (int)floorf(log2f(a));
    float frac = a / ldexpf(1.0f, e) - 1.0f;
    int m = (int)lrintf(frac * 8.0f);
    if (m == 8) {
        m = 0;
        e += 1;
    }
    if (e > 8) { // 448 = 2^8 * 1.75
        e = 8;
        m = 6;
    }
    return s | (uint8_t)((e << 3) | m);
}

// 按 [行, 每 128 列] 量化一组行主序矩阵(用于在线激活量化的 CPU 模拟)
static void quantize_rows_128(const std::vector<float> &x, int rows, int cols,
                              std::vector<uint8_t> &codes,
                              std::vector<float> &scales /* [L][rows] n-major */) {
    const int L = cols / 128;
    codes.assign((size_t)rows * cols, 0);
    scales.assign((size_t)L * rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < L; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < 128; ++i) {
                amax = fmaxf(amax, fabsf(x[(size_t)r * cols + b * 128 + i]));
            }
            const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
            scales[(size_t)b * rows + r] = s; // n-major: 沿行连续
            for (int i = 0; i < 128; ++i) {
                codes[(size_t)r * cols + b * 128 + i] = e4m3_encode(x[(size_t)r * cols + b * 128 + i] / s);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 单次探测:设置 scale 模式 -> 查 heuristic -> 跑 + 数值校验 + 计时
// ---------------------------------------------------------------------------

struct ProbeResult {
    int algos = -1;          // heuristic 返回的 algo 数(-1 = 未查询)
    bool ran = false;        // 是否真正执行
    bool pass = false;       // 抽样数值校验
    double ms = 0.0;         // 单次平均耗时
    double tflops = 0.0;
    double max_rel = 0.0;    // 抽样最大相对误差
};

#ifndef CUBLASLT_VER_MAJOR
#define CUBLASLT_VER_MAJOR 0
#define CUBLASLT_VER_MINOR 0
#define CUBLASLT_VER_PATCH 0
#endif

#if CUBLASLT_VER_MAJOR > 12 || (CUBLASLT_VER_MAJOR == 12 && CUBLASLT_VER_MINOR >= 9)
#define HAVE_BLOCK_SCALE_API 1
#else
#define HAVE_BLOCK_SCALE_API 0
#endif

static ProbeResult probe(cublasLtHandle_t lt, int M, int N, int K,
                         const std::vector<uint8_t> &w_codes,   // [N][K]
                         const std::vector<float> &w_scales,    // [N/128][L4] K-major
                         const std::vector<uint8_t> &x_codes,   // [M][K]
                         const std::vector<float> &x_scales,    // [L][M] n-major
                         int a_scale_mode, int b_scale_mode,    // cublasLtMatmulMatrixScale_t 或 -1(默认 per-tensor)
                         std::mt19937 &rng) {
    ProbeResult res;
    const int L = K / 128;
    const int L4 = (L + 3) / 4 * 4;

    uint8_t *d_w = nullptr, *d_x = nullptr;
    float *d_ws = nullptr, *d_xs = nullptr;
    __nv_bfloat16 *d_out = nullptr;
    void *d_workspace = nullptr;
    const size_t ws_bytes = 32ull << 20;
    CK_CUDA(cudaMalloc(&d_w, (size_t)N * K));
    CK_CUDA(cudaMalloc(&d_x, (size_t)M * K));
    CK_CUDA(cudaMalloc(&d_ws, w_scales.size() * sizeof(float)));
    CK_CUDA(cudaMalloc(&d_xs, x_scales.size() * sizeof(float)));
    CK_CUDA(cudaMalloc(&d_out, (size_t)M * N * sizeof(__nv_bfloat16)));
    CK_CUDA(cudaMalloc(&d_workspace, ws_bytes));
    CK_CUDA(cudaMemcpy(d_w, w_codes.data(), (size_t)N * K, cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemcpy(d_x, x_codes.data(), (size_t)M * K, cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemcpy(d_ws, w_scales.data(), w_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK_CUDA(cudaMemcpy(d_xs, x_scales.data(), x_scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, d_layout = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    CK_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    CK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    CK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

#if HAVE_BLOCK_SCALE_API
    if (a_scale_mode >= 0) {
        const cublasLtMatmulMatrixScale_t am = (cublasLtMatmulMatrixScale_t)a_scale_mode;
        CK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &am, sizeof(am)));
        CK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_ws, sizeof(d_ws)));
    }
    if (b_scale_mode >= 0) {
        const cublasLtMatmulMatrixScale_t bm = (cublasLtMatmulMatrixScale_t)b_scale_mode;
        CK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &bm, sizeof(bm)));
        CK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_xs, sizeof(d_xs)));
    }
#else
    (void)a_scale_mode;
    (void)b_scale_mode;
#endif

    // 列主序: A 存 K x N(ld=K, OP_T -> N x K), B 存 K x M(ld=K, OP_N), D 存 N x M(ld=N)
    CK_LT(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_8F_E4M3, K, N, K));
    CK_LT(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_8F_E4M3, K, M, K));
    CK_LT(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, N, M, N));
    CK_LT(cublasLtMatmulPreferenceCreate(&pref));
    CK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heur[8];
    int count = 0;
    CK_LT(cublasLtMatmulAlgoGetHeuristic(lt, desc, a_layout, b_layout, d_layout, d_layout,
                                         pref, 8, heur, &count));
    res.algos = count;
    if (count == 0) {
        goto done;
    }

    {
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t st = cublasLtMatmul(lt, desc, &alpha,
                                                 d_w, a_layout, d_x, b_layout, &beta,
                                                 d_out, d_layout, d_out, d_layout,
                                                 &heur[0].algo, d_workspace, ws_bytes, nullptr);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("  cublasLtMatmul failed: status %d\n", (int)st);
            goto done;
        }
        CK_CUDA(cudaDeviceSynchronize());
        res.ran = true;
    }

    // 抽样数值校验:随机 512 个 (m, n),CPU 双精度点积
    {
        std::vector<__nv_bfloat16> out((size_t)M * N);
        CK_CUDA(cudaMemcpy(out.data(), d_out, out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::uniform_int_distribution<int> rm(0, M - 1), rn(0, N - 1);
        double max_rel = 0.0;
        for (int s = 0; s < 512; ++s) {
            const int m = rm(rng), n = rn(rng);
            double ref = 0.0;
            for (int k = 0; k < K; ++k) {
                const double wv = e4m3_decode(w_codes[(size_t)n * K + k]) * (double)w_scales[(size_t)(n / 128) * L4 + k / 128];
                const double xv = e4m3_decode(x_codes[(size_t)m * K + k]) * (double)x_scales[(size_t)(k / 128) * M + m];
                ref += wv * xv;
            }
            const double got = __bfloat162float(out[(size_t)m * N + n]);
            const double rel = fabs(got - ref) / (fabs(ref) + 1e-6);
            max_rel = fmax(max_rel, rel);
        }
        res.max_rel = max_rel;
        res.pass = max_rel < 2e-2; // bf16 输出精度 ~0.4%;布局错误会是 ~100%
    }

    // 计时
    {
        cudaEvent_t t0, t1;
        CK_CUDA(cudaEventCreate(&t0));
        CK_CUDA(cudaEventCreate(&t1));
        const float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < 20; ++i) {
            CK_LT(cublasLtMatmul(lt, desc, &alpha, d_w, a_layout, d_x, b_layout, &beta,
                                 d_out, d_layout, d_out, d_layout, &heur[0].algo,
                                 d_workspace, ws_bytes, nullptr));
        }
        CK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < 200; ++i) {
            CK_LT(cublasLtMatmul(lt, desc, &alpha, d_w, a_layout, d_x, b_layout, &beta,
                                 d_out, d_layout, d_out, d_layout, &heur[0].algo,
                                 d_workspace, ws_bytes, nullptr));
        }
        CK_CUDA(cudaEventRecord(t1));
        CK_CUDA(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        res.ms = ms / 200.0;
        res.tflops = 2.0 * M * N * K / (res.ms * 1e-3) / 1e12;
        CK_CUDA(cudaEventDestroy(t0));
        CK_CUDA(cudaEventDestroy(t1));
    }

done:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (desc) cublasLtMatmulDescDestroy(desc);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_ws);
    cudaFree(d_xs);
    cudaFree(d_out);
    cudaFree(d_workspace);
    return res;
}

int main() {
    int rt_ver = 0, drv_ver = 0;
    cudaRuntimeGetVersion(&rt_ver);
    cudaDriverGetVersion(&drv_ver);
    printf("CUDA runtime %d.%d, driver %d.%d, cublasLt headers %d.%d.%d\n",
           rt_ver / 1000, (rt_ver % 100) / 10, drv_ver / 1000, (drv_ver % 100) / 10,
           CUBLASLT_VER_MAJOR, CUBLASLT_VER_MINOR, CUBLASLT_VER_PATCH);
#if HAVE_BLOCK_SCALE_API
    printf("block-scale API: present (VEC128_32F / BLK128x128_32F)\n");
#else
    printf("block-scale API: ABSENT (need CUDA >= 12.9 headers); only per-tensor FP8 control runs\n");
#endif
    cudaDeviceProp prop;
    CK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    cublasLtHandle_t lt;
    CK_LT(cublasLtCreate(&lt));

    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uscale(0.001f, 0.02f);

    const int shapes[][2] = {{8, 4096}, {16, 4096}, {32, 4096}};
    for (const auto &sh : shapes) {
        const int M = sh[0], N = sh[1], K = 4096;
        const int L = K / 128;
        const int L4 = (L + 3) / 4 * 4;
        printf("=== M=%d N=%d K=%d ===\n", M, N, K);

        // 权重:随机 block scale + 按块量化,内存布局 [N/128][L4](K-major,列步长 L4)
        std::vector<float> w_scales((size_t)(N / 128) * L4, 0.0f);
        std::vector<float> w_fp((size_t)N * K);
        for (auto &v : w_scales) {
            v = uscale(rng);
        }
        for (size_t i = 0; i < w_fp.size(); ++i) {
            const size_t n = i / K, k = i % K;
            w_fp[i] = gauss(rng) * 0.05f;
            (void)n;
            (void)k;
        }
        std::vector<uint8_t> w_codes((size_t)N * K);
        for (size_t i = 0; i < w_fp.size(); ++i) {
            const size_t n = i / K, k = i % K;
            w_codes[i] = e4m3_encode(w_fp[i] / w_scales[(n / 128) * L4 + k / 128]);
        }

        // 激活:模拟在线量化,scale 布局 [L][M](n-major)
        std::vector<float> x_fp((size_t)M * K);
        for (auto &v : x_fp) {
            v = gauss(rng);
        }
        std::vector<uint8_t> x_codes;
        std::vector<float> x_scales;
        quantize_rows_128(x_fp, M, K, x_codes, x_scales);

        // 对照 0:per-tensor FP8(验证 FP8 本身可用)
        {
            ProbeResult r = probe(lt, M, N, K, w_codes, w_scales, x_codes, x_scales,
                                  -1, -1, rng);
            printf("  [per-tensor FP8]        algos=%d%s\n", r.algos,
                   r.algos == 0 ? "  <-- FP8 在该平台完全不可用?" : "");
        }
#if HAVE_BLOCK_SCALE_API
        // 目标组合: 权重 BLK128x128 + 激活 VEC128
        {
            ProbeResult r = probe(lt, M, N, K, w_codes, w_scales, x_codes, x_scales,
                                  (int)CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
                                  (int)CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, rng);
            printf("  [BLK128x128 x VEC128]   algos=%d", r.algos);
            if (r.ran) {
                printf(" ran=1 %s max_rel=%.4g  %.3f ms  %.2f TFLOP/s",
                       r.pass ? "PASS" : "FAIL", r.max_rel, r.ms, r.tflops);
            }
            printf("\n");
        }
        // 备选: 双方都 VEC128(权重需细化为每行 128 列一个 scale,仅探性能上限)
        {
            // 复用同一权重码,但 scale 换成 VEC128 布局 [L4? 不需要] N-major [L][N]
            std::vector<float> w_scales_vec((size_t)L * N);
            for (int b = 0; b < L; ++b) {
                for (int n = 0; n < N; ++n) {
                    w_scales_vec[(size_t)b * N + n] = w_scales[(size_t)(n / 128) * L4 + b];
                }
            }
            ProbeResult r = probe(lt, M, N, K, w_codes, w_scales_vec, x_codes, x_scales,
                                  (int)CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
                                  (int)CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, rng);
            // 注意:此组合的参考公式与 probe 内建的 BLK 参考不同,这里只看是否能
            // 跑和耗时,不做数值校验(权重参考 scale 布局不同)。
            printf("  [VEC128 x VEC128]       algos=%d", r.algos);
            if (r.ran) {
                printf(" ran=1 (数值校验跳过)  %.3f ms  %.2f TFLOP/s", r.ms, r.tflops);
            }
            printf("\n");
        }
#endif
        printf("\n");
    }

    cublasLtDestroy(lt);
    printf("spike done. 判定标准:\n");
    printf("  - per-tensor algos=0        -> 该平台 FP8 不可用,路线否决\n");
    printf("  - block-scale algos=0       -> sm_120 未放行 block scaling(Hopper 限定),路线否决\n");
    printf("  - PASS 且 TFLOP/s 可观      -> 可作为 bs>=8 的 W8A8 备选路线(需在线量化激活)\n");
    return 0;
}
