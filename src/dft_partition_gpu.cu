#include "dft_partition_gpu.cuh"
#include "dft_atomGrid.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cmath>
namespace chemgrid::gpu
{
    __global__ void becke_partition_kernel_atom(
        int ia,
        const double *__restrict__ ax,
        const double *__restrict__ ay,
        const double *__restrict__ az,
        int A,
        const double *__restrict__ tpl_x,
        const double *__restrict__ tpl_y,
        const double *__restrict__ tpl_z,
        const double *__restrict__ tpl_w,
        int G,
        const double *__restrict__ a_mat,
        const double *__restrict__ Dij,
        double *__restrict__ out_w,
        int smooth_kind,
        double eps)
    {
        // 保险：你的 A<=200；如果越界就直接返回（也可用 assert）
        if (A > DFT_A_MAX)
            return;

        // 线程处理的点索引
        for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < G; k += blockDim.x * gridDim.x)
        {

            // 1) 即时平移到分子坐标
            const double x = tpl_x[k] + ax[ia];
            const double y = tpl_y[k] + ay[ia];
            const double z = tpl_z[k] + az[ia];

            // 2) 该点到所有原子距离 ri[i]，并把 s[i] 初始化为 1
            double ri_local[DFT_A_MAX];
            double s_local[DFT_A_MAX];

// 展开两三次足够；A 很小，逐元素也没问题
#pragma unroll 1
            for (int i = 0; i < A; ++i)
            {
                const double dxi = x - ax[i];
                const double dyi = y - ay[i];
                const double dzi = z - az[i];
                ri_local[i] = sqrt(dxi * dxi + dyi * dyi + dzi * dzi);
                s_local[i] = 1.0;
            }

// 3) 上三角成对更新（完全复现 CPU 版）
#pragma unroll 1
            for (int i = 0; i < A; ++i)
            {
                const double rii = ri_local[i];
                const double *a_row = a_mat + (size_t)i * A;
                const double *D_row = Dij + (size_t)i * A;
#pragma unroll 1
                for (int j = 0; j < i; ++j)
                {
                    double g = (rii - ri_local[j]) / (D_row[j] + eps);
                    g += (g * g - 1.0) * (-a_row[j]);
                    g = becke_smooth3(g);
                    s_local[i] *= 0.5 * (1.0 - g);
                    s_local[j] *= 0.5 * (1.0 + g);
                }
            }

            // 4) 列归一化
            double col = 0.0;
#pragma unroll 1
            for (int i = 0; i < A; ++i)
                col += s_local[i];
            if (col < eps)
                col = eps;

            // 5) 写出（简易版：同时写坐标；如需带宽优化可只写 out_w）
            out_w[k] = tpl_w[k] * s_local[ia] / col;
        }
    }

    __global__ void becke_partition_kernel_1kernel(
        // 原子中心 SoA
        const double *__restrict__ ax,
        const double *__restrict__ ay,
        const double *__restrict__ az,
        int A,

        // 拼接模板
        const double *__restrict__ tpl_x,
        const double *__restrict__ tpl_y,
        const double *__restrict__ tpl_z,
        const double *__restrict__ tpl_w,

        // gid -> ia 以及分段信息
        const int *__restrict__ atom_of_point,      // [G_total]
        const size_t *__restrict__ out_offset,      // [A]
        const size_t *__restrict__ atom_tpl_offset, // [A]
        const int *__restrict__ atom_tpl_len,       // [A]

        // 预计算矩阵（A×A）
        const double *__restrict__ a_mat, // [A*A], a[i*A+j]
        const double *__restrict__ Dij,   // [A*A], D[i*A+j]

        // 输出
        double *__restrict__ out_w, // [G_total]

        // 其它
        int smooth_kind, // 0=Original, 1=ALT（此处仅保留接口，分支由 becke_smooth3 统一）
        double eps,
        size_t G_total)
    {
        if (A > DFT_A_MAX)
            return;

        for (size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
             gid < G_total;
             gid += (size_t)blockDim.x * gridDim.x)
        {
            const int ia = atom_of_point[gid];
            const size_t ooff = out_offset[ia];
            const int k = static_cast<int>(gid - ooff);
            const size_t toff = atom_tpl_offset[ia] + (size_t)k;

            // 平移到分子坐标
            const double x = tpl_x[toff] + ax[ia];
            const double y = tpl_y[toff] + ay[ia];
            const double z = tpl_z[toff] + az[ia];

            // r_i 与 s_i
            double ri_local[DFT_A_MAX];
            double s_local[DFT_A_MAX];

#pragma unroll 1
            for (int i = 0; i < A; ++i)
            {
                const double dxi = x - ax[i];
                const double dyi = y - ay[i];
                const double dzi = z - az[i];
                ri_local[i] = sqrt(dxi * dxi + dyi * dyi + dzi * dzi);
                s_local[i] = 1.0;
            }

            // 上三角成对更新
#pragma unroll 1
            for (int i = 0; i < A; ++i)
            {
                const double rii = ri_local[i];
                const double *a_row = a_mat + (size_t)i * A;
                const double *D_row = Dij + (size_t)i * A;
#pragma unroll 1
                for (int j = 0; j < i; ++j)
                {
                    double g = (rii - ri_local[j]) / (D_row[j] + eps);
                    // 异核修正
                    g += (g * g - 1.0) * (-a_row[j]);
                    // 光滑
                    g = becke_smooth3(g);
                    // s_i, s_j 更新
                    s_local[i] *= 0.5 * (1.0 - g);
                    s_local[j] *= 0.5 * (1.0 + g);
                }
            }

            // 列归一化
            double col = 0.0;
#pragma unroll 1
            for (int i = 0; i < A; ++i)
                col += s_local[i];
            if (col < eps)
                col = eps;

            out_w[gid] = tpl_w[toff] * s_local[ia] / col;
        }
    }

}
namespace chemgrid::grid
{

    PartitionOut get_partition_from_templates_GPU(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool concat)
    {
        auto total_start = std::chrono::high_resolution_clock::now(); // 函数最开始

        // cpu侧处理
        // 0
        int A = atm_coords.size();
        // SoA 準備處理
        std::vector<double> ax(A), ay(A), az(A);
        for (int i = 0; i < A; ++i)
        {
            ax[i] = atm_coords[i][0];
            ay[i] = atm_coords[i][1];
            az[i] = atm_coords[i][2];
        }

        constexpr double eps = 1e-20;
        // 半径调整
        std::vector<double> rad(A);
        switch (radii_adjust_type)
        {
        case chemgrid::RadiiAdjustMethod::Treutler:
            for (int i = 0; i < A; ++i)
                rad[i] = std::sqrt(atomic_radii_by_Z[charges_Z[i]]) + eps;
            break;
        case chemgrid::RadiiAdjustMethod::Becke:
            for (int i = 0; i < A; ++i)
                rad[i] = atomic_radii_by_Z[charges_Z[i]] + eps;
            break;
        default:
            for (int i = 0; i < A; ++i)
                rad[i] = atomic_radii_by_Z[charges_Z[i]] + eps;
            break;
        }
        // 构造距离调整矩阵与原子距离矩阵
        std::vector<double> a_mat((size_t)A * A); // 调整距离矩阵
        std::vector<double> Dij((size_t)A * A);   // 原子距离矩阵
        for (int i = 0; i < A; ++i)
        {
            const double ri = rad[i];
            for (int j = 0; j < A; ++j)
            {
                // 调整距离矩阵
                const double rj = rad[j];
                double aij = 0.25 * ((rj / ri) - (ri / rj));
                a_mat[(size_t)i * A + j] = std::clamp(aij, -0.5, 0.5);
                // 原子距离矩阵
                const double dx = ax[i] - ax[j], dy = ay[i] - ay[j], dz = az[i] - az[j];
                Dij[(size_t)i * A + j] = std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        // 完成原子距离矩阵与调整距离矩阵构造

        // 原子轨道构造偏移
        // 先保证正序排序
        std::vector<int> z_keys;
        z_keys.reserve(grouped_by_Z.size());
        for (const auto &kv : grouped_by_Z)
            z_keys.push_back(kv.first);
        std::sort(z_keys.begin(), z_keys.end());
        // 开始构造全网格偏移

        const size_t TZ = z_keys.size();
        std::vector<size_t> tpl_offset_Z(TZ + 1, 0);
        std::vector<int> tpl_len_Z(TZ, 0);

        for (size_t t = 0; t < TZ; ++t)
        {
            tpl_len_Z[t] = static_cast<int>(grouped_by_Z.at(z_keys[t]).size());        // 该原子轨道数
            tpl_offset_Z[t + 1] = tpl_offset_Z[t] + static_cast<size_t>(tpl_len_Z[t]); // 该原子轨道数累加 到偏移量
        }
        const size_t sumGZ = tpl_offset_Z.back(); // 全部模板点数量之和

        // 把原子轨道模板点拼成方便访问的一条
        std::vector<double> tpl_x(sumGZ), tpl_y(sumGZ), tpl_z(sumGZ), tpl_w(sumGZ);
        for (size_t t = 0; t < TZ; ++t) // 遍历每种原子
        {
            const auto &tpl = grouped_by_Z.at(z_keys[t]); // vector<array<double,4>> 该原子轨道
            const size_t off = tpl_offset_Z[t];           // 偏移量
            for (size_t k = 0; k < tpl.size(); ++k)       // 遍历该原子轨道
            {
                tpl_x[off + k] = tpl[k][0]; // x
                tpl_y[off + k] = tpl[k][1]; // y
                tpl_z[off + k] = tpl[k][2]; // z
                tpl_w[off + k] = tpl[k][3]; // w
            }
        }

        // 4) 建立 Z -> (offset,len) 映射
        struct OffLen
        {
            size_t off;
            int len;
        };
        std::unordered_map<int, OffLen> Z2offlen;
        Z2offlen.reserve(TZ);
        for (size_t t = 0; t < TZ; ++t)
        {
            Z2offlen.emplace(z_keys[t], OffLen{tpl_offset_Z[t], tpl_len_Z[t]});
        }
        // 5) 遍历原子，构造 atom_tpl_offset/len + out_offset + G_total
        std::vector<size_t> atom_tpl_offset(A), out_offset(A);
        std::vector<int> atom_tpl_len(A);
        size_t G_total = 0;
        for (int ia = 0; ia < A; ++ia)
        {
            const int Z = charges_Z[ia];
            auto it = Z2offlen.find(Z);
            if (it == Z2offlen.end())
            {
                // 没有该 Z 的模板：长度置 0，但仍设置 out_offset（保持连续）
                atom_tpl_offset[ia] = 0;
                atom_tpl_len[ia] = 0;
                out_offset[ia] = G_total;
                continue;
            }
            atom_tpl_offset[ia] = it->second.off;
            atom_tpl_len[ia] = it->second.len;
            out_offset[ia] = G_total;
            G_total += static_cast<size_t>(it->second.len);
        }
        // 到这里：tpl_x/y/z/w + atom_tpl_offset/len + out_offset + G_total 都齐了

        // GPU化准备，内存开辟
        double *d_ax = 0, *d_ay = 0, *d_az = 0, *d_a_mat = 0, *d_Dij = 0;
        double *d_tpl_x = 0, *d_tpl_y = 0, *d_tpl_z = 0, *d_tpl_w = 0;
        double *d_out_w = 0;

        cudaMalloc(&d_ax, A * sizeof(double));
        cudaMalloc(&d_ay, A * sizeof(double));
        cudaMalloc(&d_az, A * sizeof(double));
        cudaMalloc(&d_a_mat, (size_t)A * A * sizeof(double));
        cudaMalloc(&d_Dij, (size_t)A * A * sizeof(double));
        cudaMalloc(&d_tpl_x, sumGZ * sizeof(double));
        cudaMalloc(&d_tpl_y, sumGZ * sizeof(double));
        cudaMalloc(&d_tpl_z, sumGZ * sizeof(double));
        cudaMalloc(&d_tpl_w, sumGZ * sizeof(double));
        cudaMalloc(&d_out_w, G_total * sizeof(double));

        cudaMemcpy(d_ax, ax.data(), A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ay, ay.data(), A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_az, az.data(), A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_mat, a_mat.data(), (size_t)A * A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Dij, Dij.data(), (size_t)A * A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tpl_x, tpl_x.data(), sumGZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tpl_y, tpl_y.data(), sumGZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tpl_z, tpl_z.data(), sumGZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tpl_w, tpl_w.data(), sumGZ * sizeof(double), cudaMemcpyHostToDevice);

        // 准备开启核函数
        const int smooth_kind = (becke_scheme == chemgrid::BeckeScheme::Original) ? 0 : 1;
        // 线程块大小：64~128 较稳（每线程有 ri/s 的小数组）
        const int TPB = 128;

        for (int ia = 0; ia < A; ++ia)
        {
            const int G = atom_tpl_len[ia];
            if (G == 0)
                continue;
            const int grid = (G + TPB - 1) / TPB;
            const size_t toff = atom_tpl_offset[ia];
            const size_t ooff = out_offset[ia];

            chemgrid::gpu::becke_partition_kernel_atom<<<grid, TPB>>>(
                ia,
                d_ax, d_ay, d_az, A,
                d_tpl_x + toff, d_tpl_y + toff, d_tpl_z + toff, d_tpl_w + toff, G,
                d_a_mat, d_Dij,
                d_out_w + ooff,
                smooth_kind, eps);
        }
        // 等待全部 kernel 结束（简单版，先用同步）
        cudaDeviceSynchronize();

        // 读回结果
        std::vector<double> weights_all(G_total);
        cudaMemcpy(weights_all.data(), d_out_w, G_total * sizeof(double), cudaMemcpyDeviceToHost);

        std::vector<std::array<double, 3>> coords_all(G_total);
        for (int ia = 0; ia < A; ++ia)
        {
            const size_t toff = atom_tpl_offset[ia], ooff = out_offset[ia];
            const int G = atom_tpl_len[ia];
            for (int k = 0; k < G; ++k)
            {
                coords_all[ooff + k] = {
                    tpl_x[toff + k] + ax[ia],
                    tpl_y[toff + k] + ay[ia],
                    tpl_z[toff + k] + az[ia]};
            }
        }

        // 释放显存（注意没有 d_out_x/y/z）
        cudaFree(d_ax);
        cudaFree(d_ay);
        cudaFree(d_az);
        cudaFree(d_a_mat);
        cudaFree(d_Dij);
        cudaFree(d_tpl_x);
        cudaFree(d_tpl_y);
        cudaFree(d_tpl_z);
        cudaFree(d_tpl_w);
        cudaFree(d_out_w);
        auto total_end = std::chrono::high_resolution_clock::now(); // return 之前
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        double total_ms = duration.count() / 1000.0;
        std::cout << "│ Total time:        " << total_ms << " ms │" << std::endl;
        return {std::move(coords_all), std::move(weights_all)};
    }

    static inline void build_global_grid_from_groupedZ(
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        const std::vector<std::array<double, 3>> &atm_coords,
        std::vector<double> &gx, std::vector<double> &gy, std::vector<double> &gz, std::vector<double> &gw,
        std::vector<int> &atom_of_point)
    {
        const int A = (int)atm_coords.size();
        size_t G_total = 0;
        for (int i = 0; i < A; ++i)
        {
            auto it = grouped_by_Z.find(charges_Z[i]);
            if (it != grouped_by_Z.end())
                G_total += it->second.size();
        }
        gx.reserve(G_total);
        gy.reserve(G_total);
        gz.reserve(G_total);
        gw.reserve(G_total);
        atom_of_point.reserve(G_total);

        for (int i = 0; i < A; ++i)
        {
            const double cx = atm_coords[i][0];
            const double cy = atm_coords[i][1];
            const double cz = atm_coords[i][2];
            auto it = grouped_by_Z.find(charges_Z[i]);
            if (it == grouped_by_Z.end())
                continue;
            const auto &tpl = it->second; // local template: (x,y,z,w) relative to atom center
            for (const auto &p : tpl)
            {
                gx.push_back(cx + p[0]);
                gy.push_back(cy + p[1]);
                gz.push_back(cz + p[2]);
                gw.push_back(p[3]);
                atom_of_point.push_back(i); // owner
            }
        }
    }

    // ====== 配置参数 ======
#ifndef TILEI
#define TILEI 16 // i 方向 tile 大小（可 32/48/64）
#endif
#ifndef TILEJ
#define TILEJ 16 // j 方向 tile 大小（可 32/48/64）
#endif
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _e = (call);                                                                   \
        if (_e != cudaSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::fflush(stderr);                                                                   \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)
#endif
    // ====== 小工具 ======
    static __device__ __forceinline__ double clampd(double x, double lo, double hi)
    {
        return x < lo ? lo : (x > hi ? hi : x);
    }
    // Horner + FMA 的三次 Becke 多项式，连用 3 次
    static __device__ __forceinline__ double becke_poly3(double x)
    {
        // 0.5*(3x - x^3) = 0.5 * x * (3 - x^2)
        return 0.5 * fma(-x * x, x, 3.0 * x);
    }
    static __device__ __forceinline__ double becke_smooth3(double x)
    {
        // 输入 x 先夹到 [-1,1]，再做三次平滑
        x = clampd(x, -1.0, 1.0);
#pragma unroll 3
        for (int t = 0; t < 3; ++t)
            x = becke_poly3(x);
        // s = (1 - x) / 2 ∈ (0,1)
        return 0.5 * (1.0 - x);
    }
    static __device__ __forceinline__ double warp_sum(double v, unsigned m = 0xffffffffu)
    {
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(m, v, off);
        return v;
    }
    static __device__ __forceinline__ double warp_max(double v, unsigned m = 0xffffffffu)
    {
        for (int off = 16; off > 0; off >>= 1)
            v = fmax(v, __shfl_xor_sync(m, v, off));
        return v;
    }

    // ====== 核函数：块级 tile × warp-per-point（owner-only 输出，不用 log/放缩）======
    __global__ void becke_partition_full_warp_point_ownerOnly(
        const double *__restrict__ ax,
        const double *__restrict__ ay,
        const double *__restrict__ az,
        int A,
        const double *__restrict__ Rij_inv,  // [A*A], row-major i*A + j
        const double *__restrict__ alpha_ij, // [A*A]
        const double *__restrict__ gx,
        const double *__restrict__ gy,
        const double *__restrict__ gz,
        const double *__restrict__ gw,         // [G] or nullptr
        const int *__restrict__ atom_of_point, // [G] owner atom index
        int G,
        double *__restrict__ out_w, // [G] final weights = gw * W_owner
        double eps)
    {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int WPB = blockDim.x >> 5;
        const int gid = blockIdx.x * WPB + warp;
        if (gid >= G)
            return;

        // 共享内存：每个 warp 切片 [r(A), P(A)]
        extern __shared__ double shmem[];
        double *r_sh = shmem + (size_t)warp * (2 * (size_t)A);
        double *P_sh = r_sh + A;

        const double x = gx[gid], y = gy[gid], z = gz[gid];
        const double wg = gw ? gw[gid] : 1.0;
        const int i_owner = atom_of_point[gid];

        // 1) r[i]
        for (int i = lane; i < A; i += 32)
        {
            const double dx = x - __ldg(&ax[i]);
            const double dy = y - __ldg(&ay[i]);
            const double dz = z - __ldg(&az[i]);
            r_sh[i] = sqrt(dx * dx + dy * dy + dz * dz);
        }
        __syncwarp();

        // 2) 对每个 i 计算 P_i = Π_{j≠i} s_ij
        //    s_ij = smooth3( clamp( (ri - rj)*Rij_inv[i,j] + alpha_ij[i,j], -1, 1 ) )
        for (int i = 0; i < A; ++i)
        {
            const double ri = r_sh[i];
            double partial_prod = 1.0;

            for (int j = lane; j < A; j += 32)
            {
                if (j == i)
                    continue;
                double mu = (ri - r_sh[j]) * __ldg(&Rij_inv[(size_t)i * A + j]); // s
                const double arg = mu + __ldg(&alpha_ij[(size_t)i * A + j]) * (1.0 - mu * mu);
                double sij = becke_smooth3(arg);
                // 防止出现 0，保持在 (eps, 1-eps) 内
                partial_prod *= sij;
            }
            // warp 乘法规约（不使用 log）
            // 用 XOR 规约：把乘法换成逐步相乘
            for (int off = 16; off > 0; off >>= 1)
            {
                partial_prod *= __shfl_xor_sync(0xffffffffu, partial_prod, off);
            }
            if (lane == 0)
                P_sh[i] = partial_prod;
            __syncwarp();
        }

        // 3) 归一化：W_i = P_i / Σ_m P_m
        double local_sum = 0.0;
        for (int i = lane; i < A; i += 32)
            local_sum += P_sh[i];
        // warp 求和规约
        for (int off = 16; off > 0; off >>= 1)
        {
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
        }
        const double sumP = local_sum; // 每个 lane 都拿到相同值

        // owner 项的权重
        // 若 sumP 极小，用 max(sumP, eps) 防止除零
        const double Wi_owner = P_sh[i_owner] / fmax(sumP, eps);
        if (lane == 0)
            out_w[gid] = Wi_owner * wg;
    }

    // ---------- helper: choose warps-per-block by shmem need ----------
    static inline int pick_WPB_for_shmem(int A)
    {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int maxOptIn = 0;
        cudaDeviceGetAttribute(&maxOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (maxOptIn <= 0)
        {
            int defSh = 0;
            cudaDeviceGetAttribute(&defSh, cudaDevAttrMaxSharedMemoryPerBlock, device);
            maxOptIn = (defSh > 0 ? defSh : 48 * 1024);
        }
        const size_t perWarp = (size_t)2 * (size_t)A * sizeof(double);
        int WPB = (int)(maxOptIn / perWarp);
        if (WPB <= 0)
            WPB = 1;
        if (WPB > 8)
            WPB = 8; // up to 8 warps/block (256 threads)
        return WPB;
    }

    static inline void launch_becke_partition_ownerOnly(
        const double *d_ax, const double *d_ay, const double *d_az, int A,
        const double *d_Rinv, const double *d_alpha,
        const double *d_gx, const double *d_gy, const double *d_gz,
        const double *d_gw,
        const int *d_atom_of_point,
        int G,
        double *d_out_w,
        double eps = 1e-20,
        cudaStream_t stream = 0)
    {
        const int WPB = pick_WPB_for_shmem(A);
        const int TPB = WPB * 32;
        const int blocks = (G + WPB - 1) / WPB;
        const size_t shmem_bytes = (size_t)WPB * (size_t)(2 * A) * sizeof(double);

        // CUDA_CHECK(cudaFuncSetAttribute(
        //     becke_partition_full_warp_point_ownerOnly,
        //     cudaFuncAttributeMaxDynamicSharedMemorySize,
        //     (int)shmem_bytes));
        becke_partition_full_warp_point_ownerOnly<<<blocks, TPB, shmem_bytes, stream>>>(
            d_ax, d_ay, d_az, A,
            d_Rinv, d_alpha,
            d_gx, d_gy, d_gz,
            d_gw,
            d_atom_of_point,
            G,
            d_out_w,
            eps);
#ifdef DEBUG
        CUDA_CHECK(cudaGetLastError());
#endif
    }

    PartitionOut get_partition_from_templates_GPU_warp_ownerOnly(
        const std::vector<std::array<double, 3>> &atm_coords,                            // A×3
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z, // templates
        const std::vector<int> &charges_Z,                                               // A
        const double *atomic_radii_by_Z                                                  // [Zmax+1] or nullptr
    )
    {
        auto total_start = std::chrono::high_resolution_clock::now(); // 函数最开始
        const int A = (int)atm_coords.size();

        // 1) atoms SoA
        std::vector<double> ax(A), ay(A), az(A);
        for (int i = 0; i < A; ++i)
        {
            ax[i] = atm_coords[i][0];
            ay[i] = atm_coords[i][1];
            az[i] = atm_coords[i][2];
        }

        // 2) pair matrices: Rij_inv (1/R) and alpha_ij (size correction)
        std::vector<double> rad(A, 1.0);
        if (atomic_radii_by_Z)
        {
            for (int i = 0; i < A; ++i)
                rad[i] = std::sqrt(atomic_radii_by_Z[charges_Z[i]]) + 1e-20;
        }
        std::vector<double> Rij_inv((size_t)A * A, 0.0);
        std::vector<double> alpha_ij((size_t)A * A, 0.0);
        for (int i = 0; i < A; ++i)
        {
            for (int j = i + 1; j < A; ++j)
            {
                const double dx = ax[i] - ax[j], dy = ay[i] - ay[j], dz = az[i] - az[j];
                const double Rij = std::sqrt(dx * dx + dy * dy + dz * dz) + 1e-20;
                const double invR = 1.0 / Rij;
                double aij = 0.25 * ((rad[j] / rad[i]) - (rad[i] / rad[j])); // anti-symmetric
                aij = std::clamp(aij, -0.5, 0.5);
                Rij_inv[(size_t)i * A + j] = invR;
                Rij_inv[(size_t)j * A + i] = invR;
                alpha_ij[(size_t)i * A + j] = aij;
                alpha_ij[(size_t)j * A + i] = -aij;
            }
        }

        // 3) build global grid & owner index
        std::vector<double> gx, gy, gz, gw;
        std::vector<int> atom_of_point;

        build_global_grid_from_groupedZ(grouped_by_Z, charges_Z, atm_coords, gx, gy, gz, gw, atom_of_point);
        const size_t G = gx.size();
        if (G == 0)
            return {};

        // 4) H2D
        double *d_ax = nullptr, *d_ay = nullptr, *d_az = nullptr;
        double *d_Rinv = nullptr, *d_alpha = nullptr;
        double *d_gx = nullptr, *d_gy = nullptr, *d_gz = nullptr, *d_gw = nullptr;
        int *d_owner = nullptr;
        double *d_out_w = nullptr;
        cudaMalloc((void **)&d_ax, A * sizeof(double));
        cudaMalloc((void **)&d_ay, A * sizeof(double));
        cudaMalloc((void **)&d_az, A * sizeof(double));
        cudaMemcpy(d_ax, ax.data(), A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ay, ay.data(), A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_az, az.data(), A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_Rinv, (size_t)A * A * sizeof(double));
        cudaMalloc((void **)&d_alpha, (size_t)A * A * sizeof(double));
        cudaMemcpy(d_Rinv, Rij_inv.data(), (size_t)A * A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha_ij.data(), (size_t)A * A * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_gx, G * sizeof(double));
        cudaMalloc((void **)&d_gy, G * sizeof(double));
        cudaMalloc((void **)&d_gz, G * sizeof(double));
        cudaMalloc((void **)&d_gw, G * sizeof(double));
        cudaMemcpy(d_gx, gx.data(), G * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gy, gy.data(), G * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gz, gz.data(), G * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gw, gw.data(), G * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_owner, G * sizeof(int));
        cudaMemcpy(d_owner, atom_of_point.data(), G * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_out_w, G * sizeof(double));
        // 5) launch
        launch_becke_partition_ownerOnly(
            d_ax, d_ay, d_az, A,
            d_Rinv, d_alpha,
            d_gx, d_gy, d_gz,
            d_gw,
            d_owner,
            (int)G,
            d_out_w,
            1e-20, 0);
        cudaDeviceSynchronize();

        // // 6) D2H & pack
        std::vector<std::array<double, 3>> coords_all(G);
        for (size_t g = 0; g < G; ++g)
            coords_all[g] = {gx[g], gy[g], gz[g]};
        std::vector<double> weights_all(G);
        cudaMemcpy(weights_all.data(), d_out_w, G * sizeof(double), cudaMemcpyDeviceToHost);
        // 结束计时
        // // 7) cleanup
        cudaFree(d_ax);
        cudaFree(d_ay);
        cudaFree(d_az);
        cudaFree(d_Rinv);
        cudaFree(d_alpha);
        cudaFree(d_gx);
        cudaFree(d_gy);
        cudaFree(d_gz);
        cudaFree(d_gw);
        cudaFree(d_owner);
        cudaFree(d_out_w);
        // 计算经过的时间
        auto total_end = std::chrono::high_resolution_clock::now(); // return 之前
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        double total_ms = duration.count() / 1000.0;
        std::cout << "│ Total time:        " << total_ms << " ms │" << std::endl;

        return {std::move(coords_all), std::move(weights_all)};
    }

}
