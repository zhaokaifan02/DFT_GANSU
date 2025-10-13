#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <dft_atomGrid.hpp>
#ifndef DFT_A_MAX
#define DFT_A_MAX 256
#endif
#ifndef DFT_M_MAX
#define DFT_M_MAX 64 // 1 + Mf\ix <= 16：支持最多 15 个邻居
#endif

// 空间块结构定义
struct SpatialBox
{
    float3 min_corner;     // 块的最小角点
    float3 max_corner;     // 块的最大角点
    int nearby_atom_count; // 临近原子数量
    int point_count;       // 块内格点数量
    int atom_list_offset;  // 在全局原子列表中的偏移
    int point_list_offset; // 在全局格点列表中的偏移
};


namespace chemgrid::gpu
{
    __device__ __forceinline__ double becke_smooth3(double g)
    {
        // clamp to [-1,1]（保持 Becke 平滑迭代的稳定性）
        g = fmin(fmax(g, -1.0), 1.0);
// 3 次迭代的多项式平滑；尽量让编译器生成FMA
#pragma unroll 3
        for (int t = 0; t < 3; ++t)
        {
            // g = 1.5g - 0.5ggg;
            double g2 = g * g;
            g = __fma_rn(-0.5, g2 * g, 1.5 * g); // FMA：(-0.5)(g^3) + 1.5g
        }
        return g;
    }
    __device__ __forceinline__ double r2_of(double x, double y, double z,
                                            double ax, double ay, double az)
    {
        double dx = x - ax, dy = y - ay, dz = z - az;
        return fma(dz, dz, fma(dy, dy, dx * dx)); // dx*dx + dy*dy + dz*dz
    }

    // 每原子一个 kernel；每线程处理该原子模板中的一个格点 k
    __global__ void becke_partition_kernel_atom(
        int ia,                        // 当前原子索引
        const double *__restrict__ ax, // A
        const double *__restrict__ ay, // A
        const double *__restrict__ az, // A
        int A,

        // 传入“该原子所用模板片段”的起始指针（host 侧已加上偏移），长度为 G
        const double *__restrict__ tpl_x, // G
        const double *__restrict__ tpl_y, // G
        const double *__restrict__ tpl_z, // G
        const double *__restrict__ tpl_w, // G
        int G,

        // 预计算：行主序 A×A
        const double *__restrict__ a_mat, // A*A
        const double *__restrict__ Dij,   // A*A

        // 输出片段（host 侧已加好 out_offset）
        double *__restrict__ out_x, // G
        double *__restrict__ out_y, // G
        double *__restrict__ out_z, // G
        double *__restrict__ out_w, // G

        int smooth_kind, // 0: Original(3次), 1: Stratmann(2次)
        double eps);

    __global__ void becke_partition_kernel_atom_nbr(
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
        const double *__restrict__ a_mat, // A*A
        const double *__restrict__ Dij,   // A*A
        const int *__restrict__ nbr_off,  // A+1
        const int *__restrict__ nbr_idx,  // sum_M
        double *__restrict__ out_w,       // 只写权重（推荐）
        int smooth_kind, double eps,
        int Mfix, double tau);
    __global__ void becke_partition_kernel_atom_dyn(
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
        const double *__restrict__ a_mat, // A*A
        const double *__restrict__ Dij,   // A*A
        double *__restrict__ out_w,       // 只写权重
        int smooth_kind, double eps,
        int Kmin, double tau);
    __global__ void becke_partition_kernel_flat(
        // 原子
        const double *__restrict__ ax,
        const double *__restrict__ ay,
        const double *__restrict__ az,
        int A,
        // 模板（所有 Z 的模板拼接在一起）
        const double *__restrict__ tpl_x,
        const double *__restrict__ tpl_y,
        const double *__restrict__ tpl_z,
        const double *__restrict__ tpl_w,
        // 每个原子对应的模板在大模板数组的起点与长度
        const size_t *__restrict__ atom_tpl_offset, // A
        const int *__restrict__ atom_tpl_len,       // A
        // “分子全局点”到“归属原子/局部模板下标”的映射（长度 G_total）
        const int *__restrict__ p2atom,
        const int *__restrict__ p2k,
        // 预计算矩阵
        const double *__restrict__ a_mat, // A*A
        const double *__restrict__ Dij,   // A*A
        // 输出（长度 G_total）
        double *__restrict__ out_w,
        // 其他参数
        int smooth_kind, double eps,
        int Kmin, double tau,
        size_t G_total);
    __global__ void becke_partition_kernel_flat_shmem(
        const double *__restrict__ ax,
        const double *__restrict__ ay,
        const double *__restrict__ az,
        int A,
        const double *__restrict__ tpl_x,
        const double *__restrict__ tpl_y,
        const double *__restrict__ tpl_z,
        const double *__restrict__ tpl_w,
        const size_t *__restrict__ atom_tpl_offset, // A
        const int *__restrict__ atom_tpl_len,       // A
        const int *__restrict__ p2atom,             // G_total
        const int *__restrict__ p2k,                // G_total
        const double *__restrict__ a_mat,           // A*A
        const double *__restrict__ Dij,             // A*A
        double *__restrict__ out_w,                 // G_total
        int smooth_kind, double eps,
        int Kmin, double tau,
        size_t G_total);
    __global__ void becke_nbr_kernel_tiled(
        double *__restrict__ weight,           // [G_total] 输入为模板权重，输出为Becke加权后权重
        const double *__restrict__ coords,     // [3*G_total] SoA (x.., y.., z..)
        const double *__restrict__ atm_coords, // [3*A] SoA (ax.., ay.., az..)
        const int *__restrict__ atm_idx,       // [G_total] 每个格点所属原子 ia
        // 邻域 CSR
        const int *__restrict__ nbr_ptr, // [A+1]
        const int *__restrict__ nbr_idx, // [sumNbr]
        // 半径（用于内核现算 a_ij）
        const double *__restrict__ rad, // [A]
        int G_total, int A,
        double eps, int smooth_kind);
    __global__ void becke_nbr_kernel_tiled_atomblock(
        double *__restrict__ weight,        // [G_total]
        const double *__restrict__ coords,  // [3*G_total] SoA (x.., y.., z..)
        const double *__restrict__ atm_xyz, // [3*A] SoA (ax.., ay.., az..)
        const int *__restrict__ nbr_ptr,    // [A+1]
        const int *__restrict__ nbr_idx,    // [sumNbr]
        const double *__restrict__ rad,     // [A]
        // block 调度（长度 = 总 block 数）
        const int *__restrict__ blk_atom,     // 该 block 的中心原子 ia
        const size_t *__restrict__ blk_start, // 在 ia 的局部输出段的起始偏移（0..G_i)
        const int *__restrict__ blk_count,    // 本 block 的格点数（≤ TILE*TILE）
        // 其它
        const size_t *__restrict__ out_offset, // [A] 每原子的全局输出偏移
        size_t G_total, int A, double eps, int smooth_kind);

    // 1kernl准备
    // 单 kernel版本
    __global__ void becke_partition_kernel_fused_points(
        // 原子中心 (SoA)
        const double *__restrict__ ax,
        const double *__restrict__ ay,
        const double *__restrict__ az,
        int A,

        // 模板（所有元素类型按 Z 拼接）
        const double *__restrict__ tpl_x,
        const double *__restrict__ tpl_y,
        const double *__restrict__ tpl_z,
        const double *__restrict__ tpl_w,

        // “全局格点 gid -> 所属原子 ia”的映射
        const int *__restrict__ atom_of_point, // [G_total]
        // 每个原子的起始输出偏移与其模板起始偏移/长度
        const long long *__restrict__ out_offset,      // [A]  注意 64-bit
        const long long *__restrict__ atom_tpl_offset, // [A]
        const int *__restrict__ atom_tpl_len,          // [A]

        // Becke 所需矩阵
        const double *__restrict__ a_mat, // [A*A]
        const double *__restrict__ Dij,   // [A*A]

        // 输出
        double *__restrict__ out_w, // [G_total]

        // 参数
        int G_total,
        int smooth_kind, double eps,
        int Kmin, double tau);

    __global__ void becke_partition_kernel_fused_points_cut(
        // 原子中心 SoA
        const double *__restrict__ ax,
        const double *__restrict__ ay,
        const double *__restrict__ az,
        int A,

        // 模板（所有Z拼接）
        const double *__restrict__ tpl_x,
        const double *__restrict__ tpl_y,
        const double *__restrict__ tpl_z,
        const double *__restrict__ tpl_w,
        const double *__restrict__ tpl_r, // 预计算 √(x^2+y^2+z^2)

        // gid -> ia 及分段信息
        const int *__restrict__ atom_of_point,         // [G_total]
        const long long *__restrict__ out_offset,      // [A]
        const long long *__restrict__ atom_tpl_offset, // [A]
        const int *__restrict__ atom_tpl_len,          // [A]

        // 预计算矩阵（A×A）
        const double *__restrict__ a_mat, // [A*A], a[i*A+j]
        const double *__restrict__ Dij,   // [A*A], D[i*A+j]

        // 邻原子 CSR
        const int *__restrict__ nbr_off, // [A+1]
        const int *__restrict__ nbr_idx, // [nnz]

        // 输出
        double *__restrict__ out_w, // [G_total]

        // 其他参数
        int G_total, int smooth_kind, double eps,
        int Kmin, double tau);

    __global__ void becke_partition_kernel_fused_points_adaptive(
        // 原子中心 (SoA)
        const double *__restrict__ ax, const double *__restrict__ ay, const double *__restrict__ az, int A,
        // 模板（所有元素类型按 Z 拼接）
        const double *__restrict__ tpl_x, const double *__restrict__ tpl_y, const double *__restrict__ tpl_z, const double *__restrict__ tpl_w,
        // "全局格点 gid -> 所属原子 ia"的映射
        const int *__restrict__ atom_of_point, // [G_total]
        // 每个原子的起始输出偏移与其模板起始偏移/长度
        const long long *__restrict__ out_offset,      // [A] 注意 64-bit
        const long long *__restrict__ atom_tpl_offset, // [A]
        const int *__restrict__ atom_tpl_len,          // [A]
        // Becke 所需矩阵
        const double *__restrict__ a_mat, // [A*A], a[i*A+j]
        const double *__restrict__ Dij,   // [A*A], D[i*A+j]
        // 输出
        double *__restrict__ out_w, // [G_total]
        // 参数
        int G_total, int smooth_kind, double eps, int Kmin, double tau);

    __global__ void count_neighbors_kernel_adaptive(
        // 原子中心 (SoA)
        const double *__restrict__ ax, const double *__restrict__ ay, const double *__restrict__ az, int A,
        // 模板（所有元素类型按 Z 拼接）
        const double *__restrict__ tpl_x, const double *__restrict__ tpl_y, const double *__restrict__ tpl_z,
        // "全局格点 gid -> 所属原子 ia"的映射
        const int *__restrict__ atom_of_point, // [G_total]
        // 每个原子的起始输出偏移与其模板起始偏移/长度
        const long long *__restrict__ out_offset,      // [A]
        const long long *__restrict__ atom_tpl_offset, // [A]
        const int *__restrict__ atom_tpl_len,          // [A]
        // 参数
        int G_total, double tau, int Kmin,
        // 输出
        int *__restrict__ L_of_gid // [G_total]
    );
}
