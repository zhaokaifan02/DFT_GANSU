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

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t error = call;                                  \
        if (error != cudaSuccess)                                  \
        {                                                          \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(error));   \
        }                                                          \
    } while (0)

    // /**
    //  * @brief CUDA kernel: 评估单个 AO 在所有网格点上的值
    //  *
    //  * 每个 thread 处理一个网格点
    //  */
    // __global__ void evaluate_single_ao_kernel(
    //     double *ao_values,         // 输出 [ngrids]
    //     const double *grid_coords, // [ngrids x 3]
    //     const double *atom_coord,  // [3] 该 AO 所在原子的坐标
    //     const double *exps,        // [nprim]
    //     const double *coeffs,      // [nprim]
    //     int nprim,
    //     int lx, int ly, int lz, // 笛卡尔幂次
    //     int ngrids)
    // {
    //     int gid = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (gid >= ngrids)
    //         return;

    //     // 网格点坐标
    //     double gx = grid_coords[gid * 3 + 0];
    //     double gy = grid_coords[gid * 3 + 1];
    //     double gz = grid_coords[gid * 3 + 2];

    //     // 相对坐标
    //     double dx = gx - atom_coord[0];
    //     double dy = gy - atom_coord[1];
    //     double dz = gz - atom_coord[2];

    //     // r^2
    //     double r2 = dx * dx + dy * dy + dz * dz;

    //     // 笛卡尔部分: x^lx * y^ly * z^lz
    //     double cart_part = 1.0;
    //     for (int i = 0; i < lx; ++i)
    //         cart_part *= dx;
    //     for (int i = 0; i < ly; ++i)
    //         cart_part *= dy;
    //     for (int i = 0; i < lz; ++i)
    //         cart_part *= dz;

    //     // 对所有基元高斯求和
    //     double ao_value = 0.0;
    //     for (int p = 0; p < nprim; ++p)
    //     {
    //         double alpha = exps[p];
    //         double coeff = coeffs[p];
    //         double radial = exp(-alpha * r2);
    //         ao_value += coeff * radial;
    //     }
    //     if (lx + ly + lz == 0)
    //         ao_value *= 0.282094791773878143;
    //     if (lx + ly + lz == 1)
    //         ao_value *= 0.488602511902919921;
    //     ao_values[gid] = ao_value * cart_part;
    // }
    /**
     * @brief CUDA kernel: 评估单个 AO
     */
    __global__ void evaluate_single_ao_kernel(
        double *ao_values,         // 输出位置（已偏移到第 ao_idx 列）
        const double *grid_coords, // [ngrids x 3]
        const double *atom_coord,  // [3]
        const double *exps,        // [nprim]
        const double *coeffs,      // [nprim]
        int nprim,
        int lx, int ly, int lz,
        int ngrids,
        int nao) // stride
    {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= ngrids)
            return;

        // 网格点坐标
        double gx = grid_coords[gid * 3 + 0];
        double gy = grid_coords[gid * 3 + 1];
        double gz = grid_coords[gid * 3 + 2];

        // 相对坐标
        double dx = gx - atom_coord[0];
        double dy = gy - atom_coord[1];
        double dz = gz - atom_coord[2];

        // r^2
        double r2 = dx * dx + dy * dy + dz * dz;

        // 笛卡尔部分
        double cart_part = 1.0;
        for (int i = 0; i < lx; ++i)
            cart_part *= dx;
        for (int i = 0; i < ly; ++i)
            cart_part *= dy;
        for (int i = 0; i < lz; ++i)
            cart_part *= dz;

        // 高斯求和
        double ao_value = 0.0;
        for (int p = 0; p < nprim; ++p)
        {
            double alpha = exps[p];
            double coeff = coeffs[p];
            double radial = exp(-alpha * r2);
            ao_value += coeff * radial;
        }

        if (lx + ly + lz == 0)
            ao_value *= 0.282094791773878143;
        if (lx + ly + lz == 1)
            ao_value *= 0.488602511902919921;
        // stride 写入
        ao_values[gid * nao] = ao_value * cart_part;
    }

    /**
     * @brief GPU 评估所有 AO（使用原始指针，避免 vector 大小限制）
     *
     * @param ao_list AO 列表
     * @param atom_coords 原子坐标
     * @param grid_coords 网格点坐标
     * @param out_ao_values 输出数组指针（调用者分配）[ngrids x nao]
     * @param ngrids 网格点数量
     * @param nao AO 数量
     */
    void evaluate_aos_on_grids_gpu_raw(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao_values, // ← 输出：调用者提供的指针
        int ngrids,
        int nao)
    {
        if (nao != (int)ao_list.size())
        {
            throw std::invalid_argument("nao != ao_list.size()");
        }
        if (ngrids != (int)grid_coords.size())
        {
            throw std::invalid_argument("ngrids != grid_coords.size()");
        }

        std::cout << "评估 " << nao << " 个 AO 在 " << ngrids << " 个网格点上...\n";

        // 检查内存大小
        size_t total_size = (size_t)ngrids * (size_t)nao * sizeof(double);
        std::cout << "总内存需求: " << total_size / (1024.0 * 1024.0) << " MB\n";

        // ========== 分配设备内存 ==========

        double *d_ao_values;
        CUDA_CHECK(cudaMalloc(&d_ao_values, total_size));
        CUDA_CHECK(cudaMemset(d_ao_values, 0, total_size)); // 初始化为 0

        // 网格坐标
        double *h_flat_grids = new double[ngrids * 3];
        for (int i = 0; i < ngrids; ++i)
        {
            h_flat_grids[i * 3 + 0] = grid_coords[i][0];
            h_flat_grids[i * 3 + 1] = grid_coords[i][1];
            h_flat_grids[i * 3 + 2] = grid_coords[i][2];
        }

        double *d_grid_coords;
        CUDA_CHECK(cudaMalloc(&d_grid_coords, ngrids * 3 * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_grid_coords, h_flat_grids,
                              ngrids * 3 * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_flat_grids;

        // kernel 配置
        int block_size = 256;
        int grid_size = (ngrids + block_size - 1) / block_size;

        // ========== loop AO ==========
        cudaEvent_t start, stop;
        float milliseconds = 0;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int ao_idx = 0; ao_idx < nao; ++ao_idx)
        {
            const AODesc &ao = ao_list[ao_idx];
            int nprim = ao.exps.size();
            // atom
            double atom_coord[3] = {
                atom_coords[ao.atom][0],
                atom_coords[ao.atom][1],
                atom_coords[ao.atom][2]};
            double *d_atom_coord;
            CUDA_CHECK(cudaMalloc(&d_atom_coord, 3 * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_atom_coord, atom_coord, 3 * sizeof(double),
                                  cudaMemcpyHostToDevice));
            // exps  coeffs
            double *d_exps;
            double *d_coeffs;
            CUDA_CHECK(cudaMalloc(&d_exps, nprim * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_coeffs, nprim * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_exps, ao.exps.data(), nprim * sizeof(double),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_coeffs, ao.coeffs.data(), nprim * sizeof(double),
                                  cudaMemcpyHostToDevice));
            // kernel start
            evaluate_single_ao_kernel<<<grid_size, block_size>>>(
                d_ao_values + ao_idx, // ao_idx offset
                d_grid_coords,
                d_atom_coord,
                d_exps,
                d_coeffs,
                nprim,
                ao.lx, ao.ly, ao.lz,
                ngrids,
                nao);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(d_atom_coord));
            CUDA_CHECK(cudaFree(d_exps));
            CUDA_CHECK(cudaFree(d_coeffs));

            if ((ao_idx + 1) % 10 == 0 || ao_idx == nao - 1)
            {
                std::cout << "  已完成 " << (ao_idx + 1) << "/" << nao << " 个 AO\n";
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "========================================" << std::endl;
        std::cout << "Total loop time: " << milliseconds << " ms" << std::endl;
        std::cout << "========================================" << std::endl;
        // 销毁事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        // --- 测速结束 ---
        // ========== 单次 memcpy 拷贝所有数据 ==========

        std::cout << "拷贝结果回 CPU (" << total_size / (1024.0 * 1024.0) << " MB)...\n";
        CUDA_CHECK(cudaMemcpy(out_ao_values, d_ao_values, total_size, cudaMemcpyDeviceToHost));

        // 释放设备内存
        CUDA_CHECK(cudaFree(d_grid_coords));
        CUDA_CHECK(cudaFree(d_ao_values));

        std::cout << "GPU 评估完成！\n";
    }

    /**
     * @brief 兼容性包装：返回 vector（小数据量时使用）
     */
    std::vector<double> evaluate_aos_on_grids_gpu(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords)
    {
        int ngrids = grid_coords.size();
        int nao = ao_list.size();

        // 检查大小是否超过 vector 限制
        size_t total_elements = (size_t)ngrids * (size_t)nao;
        size_t max_vector_size = std::vector<double>().max_size();

        if (total_elements > max_vector_size)
        {
            throw std::runtime_error(
                "数据量太大，无法用 vector 存储！请使用 evaluate_aos_on_grids_gpu_raw()");
        }

        std::vector<double> ao_values(total_elements);
        evaluate_aos_on_grids_gpu_raw(ao_list, atom_coords, grid_coords,
                                      ao_values.data(), ngrids, nao);
        return ao_values;
    }
}