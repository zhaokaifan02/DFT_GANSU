#include "dft_aoEVAL.hpp"
#include <math.h>
#ifndef BLK_X
#define BLK_X 16 // g 方向
#endif
#ifndef BLK_Y
#define BLK_Y 16 // ao 方向 (mu)
#endif

namespace chemgrid::AOEval
{

    // 比 std::pow 更快的幂：l≤6 足够用
    static __device__ __forceinline__ double pow_int(double x, int n)
    {
        double y = 1.0;
#pragma unroll
        for (int i = 0; i < n; ++i)
            y *= x;
        return y;
    }

    __global__ void kernel_eval_AO_AOmajor(
        AODevice AOd,
        const double *__restrict__ gx,
        const double *__restrict__ gy,
        const double *__restrict__ gz,
        int G,
        double *__restrict__ ao_out) // AO-major: [mu*G + g]
    {
        const int g0 = blockIdx.x * BLK_X + threadIdx.x; // 网格索引
        const int mu = blockIdx.y * BLK_Y + threadIdx.y; // AO 索引
        if (g0 >= G || mu >= AOd.nao)
            return;

        // AO 元数据
        const int c = AOd.center_of_ao[mu];
        const int lx = AOd.lx[mu];
        const int ly = AOd.ly[mu];
        const int lz = AOd.lz[mu];
        const int off = AOd.prim_offset[mu];
        const int npr = AOd.nprim[mu];

        // 位移与 r^2
        const double dx = gx[g0] - AOd.ax[c];
        const double dy = gy[g0] - AOd.ay[c];
        const double dz = gz[g0] - AOd.az[c];
        const double r2 = fma(dx, dx, fma(dy, dy, dz * dz));

        // 收缩径向和 Σ c_j exp(-α_j r^2)
        double s = 0.0;
#pragma unroll 1
        for (int j = 0; j < npr; ++j)
        {
            const double a = AOd.exps_pool[off + j];
            const double ccoeff = AOd.coeffs_pool[off + j];
            s += ccoeff * exp(-a * r2);
        }

        // 角向多项式
        double ang = 1.0;
        if (lx)
            ang *= pow_int(dx, lx);
        if (ly)
            ang *= pow_int(dy, ly);
        if (lz)
            ang *= pow_int(dz, lz);

        ao_out[(size_t)mu * (size_t)G + (size_t)g0] = s * ang;
    }

    cudaError_t launch_eval_AO_on_grid_AOmajor(
        const AODevice &AOd,
        const double *d_gx, const double *d_gy, const double *d_gz,
        int G, double *d_ao, cudaStream_t stream)
    {
        dim3 block(BLK_X, BLK_Y, 1);
        dim3 grid((G + BLK_X - 1) / BLK_X,
                  (AOd.nao + BLK_Y - 1) / BLK_Y, 1);
        kernel_eval_AO_AOmajor<<<grid, block, 0, stream>>>(AOd, d_gx, d_gy, d_gz, G, d_ao);
        return cudaGetLastError();
    }

}
