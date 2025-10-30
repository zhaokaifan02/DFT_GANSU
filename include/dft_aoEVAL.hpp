#pragma once
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include "dft_shell_eval.hpp" // AODesc

namespace chemgrid::AOEval
{
    /**
     * @brief 在网格点上评估所有 AO（GPU 加速）
     *
     * @param ao_list 完整的 AO 列表
     * @param atom_coords 原子坐标 [natom x 3]
     * @param grid_coords 网格点坐标 [ngrids x 3]
     * @return std::vector<double> AO 值矩阵 [ngrids x nao]，按行主序存储
     */
    std::vector<double> evaluate_aos_on_grids_gpu(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords);

    void evaluate_aos_on_grids_gpu_raw(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao_values, // ← 输出：调用者提供的指针
        int ngrids,
        int nao);
}

