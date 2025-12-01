#include "dftGrids.hpp"
namespace gansu::dft::gpu
{
    __device__ __forceinline__ double clampd(double x, double lo, double hi)
    {
        return x < lo ? lo : (x > hi ? hi : x);
    }
    // Horner + FMA 的三次 Becke 多项式，连用 3 次
    __device__ __forceinline__ double becke_poly3(double x)
    {
        // 0.5*(3x - x^3) = 0.5 * x * (3 - x^2)
        return 0.5 * fma(-x * x, x, 3.0 * x);
    }
    __device__ __forceinline__ double becke_smooth3(double x)
    {
        // 输入 x 先夹到 [-1,1]，再做三次平滑
        x = clampd(x, -1.0, 1.0);
#pragma unroll 3
        for (int t = 0; t < 3; ++t)
            x = becke_poly3(x);
        // s = (1 - x) / 2 ∈ (0,1)
        return 0.5 * (1.0 - x);
    }
    __device__ __forceinline__ double warp_sum(double v, unsigned m = 0xffffffffu)
    {
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(m, v, off);
        return v;
    }
    __device__ __forceinline__ double warp_max(double v, unsigned m = 0xffffffffu)
    {
        for (int off = 16; off > 0; off >>= 1)
            v = fmax(v, __shfl_xor_sync(m, v, off));
        return v;
    }

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
    inline int pick_WPB_for_shmem(int A)
    {
        int device = 0;
        cudaGetDevice(&device);
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
    inline void launch_becke_partition_ownerOnly(
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
        becke_partition_full_warp_point_ownerOnly<<<blocks, TPB, shmem_bytes, stream>>>(
            d_ax, d_ay, d_az, A,
            d_Rinv, d_alpha,
            d_gx, d_gy, d_gz,
            d_gw,
            d_atom_of_point,
            G,
            d_out_w,
            eps);
    }

    /**
     * @brief CUDA kernel: AO eval
     */
    __global__ void evaluate_single_ao_kernel(
        double *ao_values,         // output
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
        double gx = grid_coords[gid * 3 + 0];
        double gy = grid_coords[gid * 3 + 1];
        double gz = grid_coords[gid * 3 + 2];

        double dx = gx - atom_coord[0];
        double dy = gy - atom_coord[1];
        double dz = gz - atom_coord[2];

        // r^2
        double r2 = dx * dx + dy * dy + dz * dz;
        double cart_part = 1.0;
        for (int i = 0; i < lx; ++i)
            cart_part *= dx;
        for (int i = 0; i < ly; ++i)
            cart_part *= dy;
        for (int i = 0; i < lz; ++i)
            cart_part *= dz;

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
        ao_values[gid * nao] = ao_value * cart_part;
    }

}
namespace gansu::dft::chemgrid
{

    inline void build_global_grid_from_groupedZ(
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
    PartitionOut get_partition(
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
        gansu::dft::gpu::launch_becke_partition_ownerOnly(
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

        return {std::move(coords_all), std::move(weights_all)};
    }

    /**
     * @brief GPU evaluates all AOs (using raw pointers to avoid vector size limitations)
     *
     * @param ao_list List of AOs
     * @param atom_coords Atomic coordinates
     * @param grid_coords Grid point coordinates
     * @param out_ao_values Output array pointer (allocated by the caller) [ngrids x nao]
     * @param ngrids Number of grid points
     * @param nao Number of AOs
     */
    void evaluate_aos_on_grids_gpu_raw(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao_values, // ← Output: pointer provided by the caller
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

        std::cout << "Evaluating " << nao << " AOs on " << ngrids << " grid points...\n";

        // Check memory size
        size_t total_size = (size_t)ngrids * (size_t)nao * sizeof(double);
        std::cout << "Total memory requirement: " << total_size / (1024.0 * 1024.0) << " MB\n";

        // ========== Allocate device memory ==========

        double *d_ao_values;
        (cudaMalloc(&d_ao_values, total_size));
        (cudaMemset(d_ao_values, 0, total_size)); // Initialize to 0

        // Grid coordinates
        double *h_flat_grids = new double[ngrids * 3];
        for (int i = 0; i < ngrids; ++i)
        {
            h_flat_grids[i * 3 + 0] = grid_coords[i][0];
            h_flat_grids[i * 3 + 1] = grid_coords[i][1];
            h_flat_grids[i * 3 + 2] = grid_coords[i][2];
        }

        double *d_grid_coords;
        (cudaMalloc(&d_grid_coords, ngrids * 3 * sizeof(double)));
        (cudaMemcpy(d_grid_coords, h_flat_grids,
                    ngrids * 3 * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_flat_grids;

        // Kernel configuration
        int block_size = 256;
        int grid_size = (ngrids + block_size - 1) / block_size;

        // ========== Loop over AOs ==========
        cudaEvent_t start, stop;
        float milliseconds = 0;
        (cudaEventCreate(&start));
        (cudaEventCreate(&stop));
        (cudaEventRecord(start));
        for (int ao_idx = 0; ao_idx < nao; ++ao_idx)
        {
            const AODesc &ao = ao_list[ao_idx];
            int nprim = ao.exps.size();
            // Atom
            double atom_coord[3] = {
                atom_coords[ao.atom][0],
                atom_coords[ao.atom][1],
                atom_coords[ao.atom][2]};
            double *d_atom_coord;
            (cudaMalloc(&d_atom_coord, 3 * sizeof(double)));
            (cudaMemcpy(d_atom_coord, atom_coord, 3 * sizeof(double),
                        cudaMemcpyHostToDevice));
            // Exponents and coefficients
            double *d_exps;
            double *d_coeffs;
            (cudaMalloc(&d_exps, nprim * sizeof(double)));
            (cudaMalloc(&d_coeffs, nprim * sizeof(double)));
            (cudaMemcpy(d_exps, ao.exps.data(), nprim * sizeof(double),
                        cudaMemcpyHostToDevice));
            (cudaMemcpy(d_coeffs, ao.coeffs.data(), nprim * sizeof(double),
                        cudaMemcpyHostToDevice));
            // Kernel start
            gpu::evaluate_single_ao_kernel<<<grid_size, block_size>>>(
                d_ao_values + ao_idx, // ao_idx offset
                d_grid_coords,
                d_atom_coord,
                d_exps,
                d_coeffs,
                nprim,
                ao.lx, ao.ly, ao.lz,
                ngrids,
                nao);

            (cudaGetLastError());
            (cudaFree(d_atom_coord));
            (cudaFree(d_exps));
            (cudaFree(d_coeffs));

            if ((ao_idx + 1) % 10 == 0 || ao_idx == nao - 1)
            {
                std::cout << "  Completed " << (ao_idx + 1) << "/" << nao << " AOs\n";
            }
        }

        (cudaDeviceSynchronize());
        (cudaEventRecord(stop));
        (cudaEventSynchronize(stop));
        (cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "========================================" << std::endl;
        std::cout << "Total loop time: " << milliseconds << " ms" << std::endl;
        std::cout << "========================================" << std::endl;
        // Destroy events
        (cudaEventDestroy(start));
        (cudaEventDestroy(stop));
        // --- Timing ends ---
        // ========== Single memcpy to copy all data ==========

        std::cout << "Copying results back to CPU (" << total_size / (1024.0 * 1024.0) << " MB)...\n";
        (cudaMemcpy(out_ao_values, d_ao_values, total_size, cudaMemcpyDeviceToHost));

        // Free device memory
        (cudaFree(d_grid_coords));
        (cudaFree(d_ao_values));

        std::cout << "GPU evaluation completed!\n";
    }

    // ============================================================
    // CUDA Kernel: 处理同一角动量类型的所有 AO
    // 每个 block 处理一批 grid points，所有 AO 在循环中处理
    // ============================================================
    __global__ void evaluate_ao_group_kernel(
        double *__restrict__ ao_values,         // [nao x ngrids] 输出矩阵
        const double *__restrict__ grid_coords, // [ngrids x 3] 格点坐标
        const double *__restrict__ atom_coords, // [natoms x 3] 原子坐标
        const int *__restrict__ ao_indices,     // 该组中每个 AO 的原始索引
        const int *__restrict__ atom_indices,   // 该组中每个 AO 对应的原子
        const int *__restrict__ prim_offsets,   // primitive 偏移
        const int *__restrict__ prim_counts,    // primitive 数量
        const double *__restrict__ all_exps,    // 所有 exponents
        const double *__restrict__ all_coeffs,  // 所有 coefficients
        int lx, int ly, int lz,                 // 角动量 (该组共享)
        int num_aos_in_group,                   // 该组的 AO 数量
        int ngrids,                             // 格点总数
        int nao)                                // AO 总数 (用于输出 stride)
    {
        int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (grid_idx >= ngrids)
            return;

        // 读取当前格点坐标
        double gx = grid_coords[grid_idx * 3 + 0];
        double gy = grid_coords[grid_idx * 3 + 1];
        double gz = grid_coords[grid_idx * 3 + 2];

        // 遍历该组中的所有 AO
        for (int i = 0; i < num_aos_in_group; ++i)
        {
            int ao_idx = ao_indices[i];
            int atom_idx = atom_indices[i];

            // 原子坐标
            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            // 相对坐标
            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double r2 = dx * dx + dy * dy + dz * dz;

            // 角动量部分 (该组所有 AO 共享相同的 lx, ly, lz)
            double angular = 1.0;
            for (int ix = 0; ix < lx; ++ix)
                angular *= dx;
            for (int iy = 0; iy < ly; ++iy)
                angular *= dy;
            for (int iz = 0; iz < lz; ++iz)
                angular *= dz;

            // 径向部分：收缩高斯
            double radial = 0.0;
            int prim_start = prim_offsets[i];
            int nprim = prim_counts[i];

            for (int p = 0; p < nprim; ++p)
            {
                double alpha = all_exps[prim_start + p];
                double coeff = all_coeffs[prim_start + p];
                radial += coeff * exp(-alpha * r2);
            }

            // 写入结果: ao_values[ao_idx * ngrids + grid_idx]
            // 或者 ao_values[grid_idx * nao + ao_idx] 取决于你的存储方式
            // 这里假设是 [nao, ngrids] 布局 (ao-major)
            ao_values[ao_idx * ngrids + grid_idx] = angular * radial;
        }
    }
    // fan AOupdate
    //  ============================================================
    // Host 端函数：将 AO 按角动量类型分组
    // ============================================================
    inline std::map<AngularMomentumKey, AOGroupData> group_aos_by_angular_momentum(
        const std::vector<AODesc> &ao_list)
    {
        std::map<AngularMomentumKey, AOGroupData> groups;

        for (size_t ao_idx = 0; ao_idx < ao_list.size(); ++ao_idx)
        {
            const AODesc &ao = ao_list[ao_idx];
            AngularMomentumKey key{ao.lx, ao.ly, ao.lz};

            AOGroupData &group = groups[key];
            group.lx = ao.lx;
            group.ly = ao.ly;
            group.lz = ao.lz;

            // 记录 primitive 偏移
            int offset = group.all_exps.size();
            group.prim_offsets.push_back(offset);
            group.prim_counts.push_back(ao.exps.size());

            // 添加 AO 信息
            group.ao_indices.push_back(ao_idx);
            group.atom_indices.push_back(ao.atom);

            // 打包 exponents 和 coefficients
            for (double e : ao.exps)
                group.all_exps.push_back(e);
            for (double c : ao.coeffs)
                group.all_coeffs.push_back(c);
        }

        return groups;
    }

    // ============================================================
    // 主函数：按角动量类型分组的 GPU 评估
    // ============================================================
    void evaluate_aos_on_grids_gpu_grouped(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao_values, // [nao x ngrids] 输出
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

        std::cout << "=== Grouped GPU AO Evaluation ===" << std::endl;
        std::cout << "Evaluating " << nao << " AOs on " << ngrids << " grid points...\n";

        // Step 1: 按角动量分组
        auto groups = group_aos_by_angular_momentum(ao_list);
        std::cout << "Number of angular momentum groups: " << groups.size() << std::endl;

        for (const auto &[key, group] : groups)
        {
            std::cout << "  Group (lx=" << key.lx << ", ly=" << key.ly
                      << ", lz=" << key.lz << "): " << group.ao_indices.size()
                      << " AOs" << std::endl;
        }

        // Step 2: 分配设备内存
        size_t total_size = (size_t)ngrids * (size_t)nao * sizeof(double);
        std::cout << "Total memory requirement: " << total_size / (1024.0 * 1024.0) << " MB\n";

        double *d_ao_values;
        cudaMalloc(&d_ao_values, total_size);
        cudaMemset(d_ao_values, 0, total_size);

        // 格点坐标
        double *h_flat_grids = new double[ngrids * 3];
        for (int i = 0; i < ngrids; ++i)
        {
            h_flat_grids[i * 3 + 0] = grid_coords[i][0];
            h_flat_grids[i * 3 + 1] = grid_coords[i][1];
            h_flat_grids[i * 3 + 2] = grid_coords[i][2];
        }
        double *d_grid_coords;
        cudaMalloc(&d_grid_coords, ngrids * 3 * sizeof(double));
        cudaMemcpy(d_grid_coords, h_flat_grids, ngrids * 3 * sizeof(double), cudaMemcpyHostToDevice);
        delete[] h_flat_grids;

        // 原子坐标
        int natoms = atom_coords.size();
        double *h_atom_coords = new double[natoms * 3];
        for (int i = 0; i < natoms; ++i)
        {
            h_atom_coords[i * 3 + 0] = atom_coords[i][0];
            h_atom_coords[i * 3 + 1] = atom_coords[i][1];
            h_atom_coords[i * 3 + 2] = atom_coords[i][2];
        }
        double *d_atom_coords;
        cudaMalloc(&d_atom_coords, natoms * 3 * sizeof(double));
        cudaMemcpy(d_atom_coords, h_atom_coords, natoms * 3 * sizeof(double), cudaMemcpyHostToDevice);
        delete[] h_atom_coords;

        // Kernel 配置
        int block_size = 256;
        int grid_size = (ngrids + block_size - 1) / block_size;

        // Step 3: 计时并执行每个分组的 kernel
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int group_idx = 0;
        for (const auto &[key, group] : groups)
        {
            int num_aos = group.ao_indices.size();
            int num_prims = group.all_exps.size();

            // 计算该组最大 primitive 数量 (用于 shared memory)
            int max_prim = 0;
            for (int c : group.prim_counts)
            {
                if (c > max_prim)
                    max_prim = c;
            }

            // 分配该组的设备内存
            int *d_ao_indices, *d_atom_indices, *d_prim_offsets, *d_prim_counts;
            double *d_all_exps, *d_all_coeffs;

            cudaMalloc(&d_ao_indices, num_aos * sizeof(int));
            cudaMalloc(&d_atom_indices, num_aos * sizeof(int));
            cudaMalloc(&d_prim_offsets, num_aos * sizeof(int));
            cudaMalloc(&d_prim_counts, num_aos * sizeof(int));
            cudaMalloc(&d_all_exps, num_prims * sizeof(double));
            cudaMalloc(&d_all_coeffs, num_prims * sizeof(double));

            cudaMemcpy(d_ao_indices, group.ao_indices.data(), num_aos * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_atom_indices, group.atom_indices.data(), num_aos * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prim_offsets, group.prim_offsets.data(), num_aos * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prim_counts, group.prim_counts.data(), num_aos * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_exps, group.all_exps.data(), num_prims * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_coeffs, group.all_coeffs.data(), num_prims * sizeof(double), cudaMemcpyHostToDevice);

            // 计算 shared memory 大小
            size_t shared_mem_size = 2 * max_prim * sizeof(double);

            // 启动 kernel (使用 shared memory 优化版本)
            evaluate_ao_group_kernel<<<grid_size, block_size>>>(
                d_ao_values,
                d_grid_coords,
                d_atom_coords,
                d_ao_indices,
                d_atom_indices,
                d_prim_offsets,
                d_prim_counts,
                d_all_exps,
                d_all_coeffs,
                group.lx, group.ly, group.lz,
                num_aos,
                ngrids,
                nao);
                
            cudaGetLastError();

            // 释放该组的临时内存
            cudaFree(d_ao_indices);
            cudaFree(d_atom_indices);
            cudaFree(d_prim_offsets);
            cudaFree(d_prim_counts);
            cudaFree(d_all_exps);
            cudaFree(d_all_coeffs);

            std::cout << "  Completed group " << (group_idx + 1) << "/" << groups.size()
                      << " (lx=" << key.lx << ", ly=" << key.ly << ", lz=" << key.lz
                      << ", " << num_aos << " AOs)" << std::endl;
            group_idx++;
        }

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "========================================" << std::endl;
        std::cout << "Total kernel time: " << milliseconds << " ms" << std::endl;
        std::cout << "Number of kernel launches: " << groups.size() << std::endl;
        std::cout << "========================================" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Step 4: 复制结果回主机
        std::cout << "Copying results back to CPU (" << total_size / (1024.0 * 1024.0) << " MB)...\n";
        cudaMemcpy(out_ao_values, d_ao_values, total_size, cudaMemcpyDeviceToHost);

        // 释放设备内存
        cudaFree(d_grid_coords);
        cudaFree(d_atom_coords);
        cudaFree(d_ao_values);

        std::cout << "Grouped GPU evaluation completed!\n";
    }
}