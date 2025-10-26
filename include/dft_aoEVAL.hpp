#pragma once
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include "shell_eval.hpp" // AODesc

namespace chemgrid::AOEval
{
    // ========== Device 端 SoA 结构（只保存扁平化后的只读数据）==========
    struct AODevice
    {
        int nao;
        int natm;
        int nprim_tot;

        // AO 元数据（每个 AO 的中心与幂、以及其在 primitives 池中的 offset/长度）
        const int *center_of_ao; // [nao]
        const int *lx;           // [nao]
        const int *ly;           // [nao]
        const int *lz;           // [nao]
        const int *prim_offset;  // [nao]
        const int *nprim;        // [nao]

        // 原子坐标 SoA
        const double *ax, *ay, *az; // [natm]

        // primitives 池
        const double *exps_pool;   // [nprim_tot]
        const double *coeffs_pool; // [nprim_tot]
    };

    // ========== Host 端缓冲区（持有 device 指针，负责分配/释放）==========
    struct AODeviceBuffers
    {
        // meta
        int *d_center = nullptr, *d_lx = nullptr, *d_ly = nullptr, *d_lz = nullptr;
        int *d_off = nullptr, *d_npr = nullptr;

        // atoms
        double *d_ax = nullptr, *d_ay = nullptr, *d_az = nullptr;

        // pools
        double *d_exps = nullptr, *d_coeffs = nullptr;

        // grids
        double *d_gx = nullptr, *d_gy = nullptr, *d_gz = nullptr;

        // output
        double *d_ao = nullptr; // [nao*G], AO-major: mu*G + g

        void free_all();
    };

    // ========== 扁平化：把 std::vector<AODesc> + 原子坐标 → SoA 并上传 ==========
    struct AOFlatHost
    {
        // host side SoA for upload
        std::vector<int> center_of_ao, lx, ly, lz, prim_offset, nprim;
        std::vector<double> ax, ay, az;
        std::vector<double> exps_pool, coeffs_pool;
        int nao = 0, natm = 0, nprim_tot = 0;
    };

    AOFlatHost flatten_to_soa(const std::vector<AODesc> &aos,
                              const std::vector<std::array<double, 3>> &atm_coords);

    void upload_AO_to_device(const AOFlatHost &A, AODevice &AOd, AODeviceBuffers &buf);

    // ========== 网格上传/计算/下载 ==========
    void upload_grids_to_device(const std::vector<std::array<double, 3>> &grids,
                                AODeviceBuffers &buf);

    cudaError_t launch_eval_AO_on_grid_AOmajor( // 输出 AO-major: ao[mu*G + g]
        const AODevice &AOd,
        const double *d_gx, const double *d_gy, const double *d_gz, // [G]
        int G,
        double *d_ao, // [nao*G]
        cudaStream_t stream = 0);

    void download_ao(std::vector<double> &ao_host, int nao, int G, const AODeviceBuffers &buf);

    // ========== 方便的一站式封装（Host 入，GPU 算，Host 出）=========
    void evaluate_AO_on_grid_GPU_AOmajor(
        const std::vector<AODesc> &aos,
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::vector<std::array<double, 3>> &grids,
        std::vector<double> &ao_out); // 大小 nao*ngrids，布局: mu*G + g

}