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
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_kernel_time_ms = 0.0f;

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
            cudaEventRecord(start, 0); //
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
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_kernel_time_ms += milliseconds; //
                                                  // --------------------------------
            (cudaGetLastError());
            (cudaFree(d_atom_coord));
            (cudaFree(d_exps));
            (cudaFree(d_coeffs));

            if ((ao_idx + 1) % 10 == 0 || ao_idx == nao - 1)
            {
                std::cout << "  Completed " << (ao_idx + 1) << "/" << nao << " AOs\n";
            }
        }

        std::cout << "========================================" << std::endl;
        std::cout << " ORGINAL Total PURE Kernel Execution Time: " << total_kernel_time_ms << " ms" << std::endl;
        std::cout << "========================================" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        // -----------------------

        std::cout << "Copying results back to CPU (" << total_size / (1024.0 * 1024.0) << " MB)...\n";
        (cudaMemcpy(out_ao_values, d_ao_values, total_size, cudaMemcpyDeviceToHost));

        // Free device memory
        (cudaFree(d_grid_coords));
        (cudaFree(d_ao_values));

        std::cout << "GPU evaluation completed!\n";
    }

    /*****
     * FAN 12 3 GM
     * shell base AO evaluate
     */

    // ============================================================
    // Spherical harmonic normalization factor
    // ============================================================
    __host__ __device__ inline double get_fac(int l)
    {
        const double fac_table[] = {
            0.282094791773878143, // l=0
            0.488602511902919921, // l=1
            0.630783130505040012, // l=2
            0.746352665180230783, // l=3
            0.846284375321634481, // l=4
            0.935414346693485387, // l=5
            1.016220929899498370, // l=6
        };
        if (l >= 0 && l <= 6)
            return fac_table[l];
        return sqrt((2.0 * l + 1.0) / (4.0 * M_PI));
    }
    // ============================================================
    // Device helper: safe power
    // ============================================================
    __device__ inline double dev_power(double x, int n)
    {
        if (n == 0)
            return 1.0;
        if (n == 1)
            return x;
        if (n == 2)
            return x * x;
        if (n == 3)
            return x * x * x;
        double r = 1.0;
        for (int i = 0; i < n; ++i)
            r *= x;
        return r;
    }
    // ============================================================
    // Kernel for s orbitals (l=0)
    // Output layout: [ngrids x nao]
    // ============================================================
    __global__ void evaluate_s_shells_kernel(
        double *__restrict__ ao_values, // [ngrids x nao]
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_indices,
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid_idx >= ngrids)
            return;

        double gx = grid_coords[grid_idx * 3 + 0];
        double gy = grid_coords[grid_idx * 3 + 1];
        double gz = grid_coords[grid_idx * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            int ao_idx = shell_ao_indices[s];
            int atom_idx = shell_atom_indices[s];
            int prim_start = shell_prim_offsets[s];
            int nprim = shell_prim_counts[s];

            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                ce += all_coeffs[prim_start + p] * exp(-all_exps[prim_start + p] * r2);
            }
            ce *= fac;

            // Output: ao_values[grid_idx, ao_idx]
            ao_values[grid_idx * nao + ao_idx] = ce;
        }
    }

    // ============================================================
    // Kernel for p orbitals (l=1): 3 AOs per shell
    // Output layout: [ngrids x nao]
    // ============================================================
    __global__ void evaluate_p_shells_kernel(
        double *__restrict__ ao_values,
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start,
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid_idx >= ngrids)
            return;

        double gx = grid_coords[grid_idx * 3 + 0];
        double gy = grid_coords[grid_idx * 3 + 1];
        double gz = grid_coords[grid_idx * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            int ao_start = shell_ao_start[s];
            int atom_idx = shell_atom_indices[s];
            int prim_start = shell_prim_offsets[s];
            int nprim = shell_prim_counts[s];

            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double r2 = dx * dx + dy * dy + dz * dz;

            // Compute ce ONCE for all 3 p orbitals
            double ce = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                ce += all_coeffs[prim_start + p] * exp(-all_exps[prim_start + p] * r2);
            }
            ce *= fac;

            // Output: ao_values[grid_idx, ao_start + i]
            ao_values[grid_idx * nao + ao_start + 0] = ce * dx; // px
            ao_values[grid_idx * nao + ao_start + 1] = ce * dy; // py
            ao_values[grid_idx * nao + ao_start + 2] = ce * dz; // pz
        }
    }

    // ============================================================
    // Kernel for d orbitals (l=2): 6 AOs per shell
    // Order: xx, xy, xz, yy, yz, zz
    // ============================================================
    __global__ void evaluate_d_shells_kernel(
        double *__restrict__ ao_values,
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start,
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid_idx >= ngrids)
            return;

        double gx = grid_coords[grid_idx * 3 + 0];
        double gy = grid_coords[grid_idx * 3 + 1];
        double gz = grid_coords[grid_idx * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            int ao_start = shell_ao_start[s];
            int atom_idx = shell_atom_indices[s];
            int prim_start = shell_prim_offsets[s];
            int nprim = shell_prim_counts[s];

            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double r2 = dx * dx + dy * dy + dz * dz;

            // Compute ce ONCE
            double ce = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                ce += all_coeffs[prim_start + p] * exp(-all_exps[prim_start + p] * r2);
            }
            ce *= fac;

            double dx2 = dx * dx;
            double dy2 = dy * dy;
            double dz2 = dz * dz;

            int base = grid_idx * nao + ao_start;
            ao_values[base + 0] = ce * dx2;     // dxx
            ao_values[base + 1] = ce * dx * dy; // dxy
            ao_values[base + 2] = ce * dx * dz; // dxz
            ao_values[base + 3] = ce * dy2;     // dyy
            ao_values[base + 4] = ce * dy * dz; // dyz
            ao_values[base + 5] = ce * dz2;     // dzz
        }
    }

    // ============================================================
    // Kernel for f orbitals (l=3): 10 AOs per shell
    // ============================================================
    __global__ void evaluate_f_shells_kernel(
        double *__restrict__ ao_values,
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start,
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid_idx >= ngrids)
            return;

        double gx = grid_coords[grid_idx * 3 + 0];
        double gy = grid_coords[grid_idx * 3 + 1];
        double gz = grid_coords[grid_idx * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            int ao_start = shell_ao_start[s];
            int atom_idx = shell_atom_indices[s];
            int prim_start = shell_prim_offsets[s];
            int nprim = shell_prim_counts[s];

            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                ce += all_coeffs[prim_start + p] * exp(-all_exps[prim_start + p] * r2);
            }
            ce *= fac;

            double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
            double dx3 = dx2 * dx, dy3 = dy2 * dy, dz3 = dz2 * dz;

            int base = grid_idx * nao + ao_start;
            ao_values[base + 0] = ce * dx3;          // xxx
            ao_values[base + 1] = ce * dx2 * dy;     // xxy
            ao_values[base + 2] = ce * dx2 * dz;     // xxz
            ao_values[base + 3] = ce * dx * dy2;     // xyy
            ao_values[base + 4] = ce * dx * dy * dz; // xyz
            ao_values[base + 5] = ce * dx * dz2;     // xzz
            ao_values[base + 6] = ce * dy3;          // yyy
            ao_values[base + 7] = ce * dy2 * dz;     // yyz
            ao_values[base + 8] = ce * dy * dz2;     // yzz
            ao_values[base + 9] = ce * dz3;          // zzz
        }
    }

    // ============================================================
    // Kernel for g orbitals (l=4): 15 AOs per shell
    // ============================================================
    __global__ void evaluate_g_shells_kernel(
        double *__restrict__ ao_values,
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start,
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid_idx >= ngrids)
            return;

        double gx = grid_coords[grid_idx * 3 + 0];
        double gy = grid_coords[grid_idx * 3 + 1];
        double gz = grid_coords[grid_idx * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            int ao_start = shell_ao_start[s];
            int atom_idx = shell_atom_indices[s];
            int prim_start = shell_prim_offsets[s];
            int nprim = shell_prim_counts[s];

            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                ce += all_coeffs[prim_start + p] * exp(-all_exps[prim_start + p] * r2);
            }
            ce *= fac;

            double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
            double dx3 = dx2 * dx, dy3 = dy2 * dy, dz3 = dz2 * dz;
            double dx4 = dx2 * dx2, dy4 = dy2 * dy2, dz4 = dz2 * dz2;

            int base = grid_idx * nao + ao_start;
            ao_values[base + 0] = ce * dx4;           // xxxx
            ao_values[base + 1] = ce * dx3 * dy;      // xxxy
            ao_values[base + 2] = ce * dx3 * dz;      // xxxz
            ao_values[base + 3] = ce * dx2 * dy2;     // xxyy
            ao_values[base + 4] = ce * dx2 * dy * dz; // xxyz
            ao_values[base + 5] = ce * dx2 * dz2;     // xxzz
            ao_values[base + 6] = ce * dx * dy3;      // xyyy
            ao_values[base + 7] = ce * dx * dy2 * dz; // xyyz
            ao_values[base + 8] = ce * dx * dy * dz2; // xyzz
            ao_values[base + 9] = ce * dx * dz3;      // xzzz
            ao_values[base + 10] = ce * dy4;          // yyyy
            ao_values[base + 11] = ce * dy3 * dz;     // yyyz
            ao_values[base + 12] = ce * dy2 * dz2;    // yyzz
            ao_values[base + 13] = ce * dy * dz3;     // yzzz
            ao_values[base + 14] = ce * dz4;          // zzzz
        }
    }

    // ============================================================
    // Generic kernel for higher angular momentum (l >= 5)
    // ============================================================
    __global__ void evaluate_general_shells_kernel(
        double *__restrict__ ao_values,
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start,
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const int *__restrict__ shell_ncomponents,
        const int *__restrict__ all_lx,
        const int *__restrict__ all_ly,
        const int *__restrict__ all_lz,
        const int *__restrict__ ao_component_offsets,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid_idx >= ngrids)
            return;

        double gx = grid_coords[grid_idx * 3 + 0];
        double gy = grid_coords[grid_idx * 3 + 1];
        double gz = grid_coords[grid_idx * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            int ao_start = shell_ao_start[s];
            int atom_idx = shell_atom_indices[s];
            int prim_start = shell_prim_offsets[s];
            int nprim = shell_prim_counts[s];
            int ncomp = shell_ncomponents[s];
            int comp_offset = ao_component_offsets[s];

            double ax = atom_coords[atom_idx * 3 + 0];
            double ay = atom_coords[atom_idx * 3 + 1];
            double az = atom_coords[atom_idx * 3 + 2];

            double dx = gx - ax;
            double dy = gy - ay;
            double dz = gz - az;
            double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                ce += all_coeffs[prim_start + p] * exp(-all_exps[prim_start + p] * r2);
            }
            ce *= fac;

            int base = grid_idx * nao + ao_start;
            for (int c = 0; c < ncomp; ++c)
            {
                int lx = all_lx[comp_offset + c];
                int ly = all_ly[comp_offset + c];
                int lz = all_lz[comp_offset + c];
                double angular = dev_power(dx, lx) * dev_power(dy, ly) * dev_power(dz, lz);
                ao_values[base + c] = ce * angular;
            }
        }
    }

    // fan AOupdate
    // ============================================================
    // Host: Group AOs by shell
    // ============================================================
    inline std::map<int, std::vector<ShellData>> group_aos_by_shell(
        const std::vector<AODesc> &ao_list)
    {
        std::map<int, std::vector<ShellData>> shell_groups;

        int nao = ao_list.size();
        int i = 0;

        while (i < nao)
        {
            const AODesc &first_ao = ao_list[i];
            ShellData shell;
            shell.atom_idx = first_ao.atom;
            shell.l = first_ao.l;
            shell.fac = get_fac(first_ao.l);
            shell.exps = first_ao.exps;
            shell.coeffs = first_ao.coeffs;
            shell.nprim = first_ao.exps.size();

            shell.ao_indices.push_back(i);
            shell.lx_list.push_back(first_ao.lx);
            shell.ly_list.push_back(first_ao.ly);
            shell.lz_list.push_back(first_ao.lz);

            int j = i + 1;
            while (j < nao)
            {
                const AODesc &ao = ao_list[j];
                if (ao.atom == first_ao.atom &&
                    ao.l == first_ao.l &&
                    ao.exps == first_ao.exps &&
                    ao.coeffs == first_ao.coeffs)
                {
                    shell.ao_indices.push_back(j);
                    shell.lx_list.push_back(ao.lx);
                    shell.ly_list.push_back(ao.ly);
                    shell.lz_list.push_back(ao.lz);
                    j++;
                }
                else
                {
                    break;
                }
            }

            shell_groups[shell.l].push_back(shell);
            i = j;
        }

        return shell_groups;
    }
    // ============================================================
    // Main function: Shell-grouped GPU AO evaluation
    // ============================================================
    void evaluate_aos_gpu_shell_grouped(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao_values,
        int ngrids,
        int nao)
    {
        auto shell_groups = group_aos_by_shell(ao_list);

        std::cout << "Shell-grouped AO evaluation (grid-major output):" << std::endl;
        for (const auto &[l, shells] : shell_groups)
        {
            int total_aos = 0;
            for (const auto &s : shells)
                total_aos += s.ao_indices.size();
            std::cout << "  l=" << l << ": " << shells.size() << " shells, "
                      << total_aos << " AOs" << std::endl;
        }

        // Allocate device memory
        size_t total_size = (size_t)ngrids * nao * sizeof(double);
        double *d_ao_values;
        cudaMalloc(&d_ao_values, total_size);
        cudaMemset(d_ao_values, 0, total_size);

        // Grid coordinates [ngrids x 3]
        std::vector<double> h_flat_grids(ngrids * 3);
        for (int i = 0; i < ngrids; ++i)
        {
            h_flat_grids[i * 3 + 0] = grid_coords[i][0];
            h_flat_grids[i * 3 + 1] = grid_coords[i][1];
            h_flat_grids[i * 3 + 2] = grid_coords[i][2];
        }
        double *d_grid_coords;
        cudaMalloc(&d_grid_coords, ngrids * 3 * sizeof(double));
        cudaMemcpy(d_grid_coords, h_flat_grids.data(), ngrids * 3 * sizeof(double), cudaMemcpyHostToDevice);

        // Atom coordinates [natoms x 3]
        int natoms = atom_coords.size();
        std::vector<double> h_atom_coords(natoms * 3);
        for (int i = 0; i < natoms; ++i)
        {
            h_atom_coords[i * 3 + 0] = atom_coords[i][0];
            h_atom_coords[i * 3 + 1] = atom_coords[i][1];
            h_atom_coords[i * 3 + 2] = atom_coords[i][2];
        }
        double *d_atom_coords;
        cudaMalloc(&d_atom_coords, natoms * 3 * sizeof(double));
        cudaMemcpy(d_atom_coords, h_atom_coords.data(), natoms * 3 * sizeof(double), cudaMemcpyHostToDevice);

        int block_size = 256;
        int grid_size = (ngrids + block_size - 1) / block_size;
        // --- [新增] 初始化 CUDA 事件用于计时 ---
        cudaEvent_t start, stop;           // <--- 声明事件
        cudaEventCreate(&start);           // <--- 创建事件
        cudaEventCreate(&stop);            // <--- 创建事件
        float total_kernel_time_ms = 0.0f; // <--- 用于累加所有 kernel 的时间
                                           // -------------------------------------
        // Process each angular momentum group
        for (const auto &[l, shells] : shell_groups)
        {
            int num_shells = shells.size();
            double fac = get_fac(l);

            // Prepare shell data
            std::vector<int> h_ao_start, h_atom_indices, h_prim_offsets, h_prim_counts;
            std::vector<double> h_all_exps, h_all_coeffs;

            int prim_offset = 0;
            for (const auto &shell : shells)
            {
                h_ao_start.push_back(shell.ao_indices[0]);
                h_atom_indices.push_back(shell.atom_idx);
                h_prim_offsets.push_back(prim_offset);
                h_prim_counts.push_back(shell.nprim);

                for (double e : shell.exps)
                    h_all_exps.push_back(e);
                for (double c : shell.coeffs)
                    h_all_coeffs.push_back(c);
                prim_offset += shell.nprim;
            }

            // Allocate and copy
            int *d_ao_start, *d_atom_indices, *d_prim_offsets, *d_prim_counts;
            double *d_all_exps, *d_all_coeffs;

            cudaMalloc(&d_ao_start, num_shells * sizeof(int));
            cudaMalloc(&d_atom_indices, num_shells * sizeof(int));
            cudaMalloc(&d_prim_offsets, num_shells * sizeof(int));
            cudaMalloc(&d_prim_counts, num_shells * sizeof(int));
            cudaMalloc(&d_all_exps, h_all_exps.size() * sizeof(double));
            cudaMalloc(&d_all_coeffs, h_all_coeffs.size() * sizeof(double));

            cudaMemcpy(d_ao_start, h_ao_start.data(), num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_atom_indices, h_atom_indices.data(), num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prim_offsets, h_prim_offsets.data(), num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prim_counts, h_prim_counts.data(), num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_exps, h_all_exps.data(), h_all_exps.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_coeffs, h_all_coeffs.data(), h_all_coeffs.size() * sizeof(double), cudaMemcpyHostToDevice);

            // Launch appropriate kernel
            cudaEventRecord(start, 0); // <--- 在 Kernel 启动前记录 start
            if (l == 0)
            {
                evaluate_s_shells_kernel<<<grid_size, block_size>>>(
                    d_ao_values, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else if (l == 1)
            {
                evaluate_p_shells_kernel<<<grid_size, block_size>>>(
                    d_ao_values, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else if (l == 2)
            {
                evaluate_d_shells_kernel<<<grid_size, block_size>>>(
                    d_ao_values, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else if (l == 3)
            {
                evaluate_f_shells_kernel<<<grid_size, block_size>>>(
                    d_ao_values, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else if (l == 4)
            {
                evaluate_g_shells_kernel<<<grid_size, block_size>>>(
                    d_ao_values, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else
            {
                // General case for l >= 5
                std::vector<int> h_ncomponents, h_comp_offsets;
                std::vector<int> h_all_lx, h_all_ly, h_all_lz;

                int comp_offset = 0;
                for (const auto &shell : shells)
                {
                    h_ncomponents.push_back(shell.ao_indices.size());
                    h_comp_offsets.push_back(comp_offset);
                    for (size_t c = 0; c < shell.ao_indices.size(); ++c)
                    {
                        h_all_lx.push_back(shell.lx_list[c]);
                        h_all_ly.push_back(shell.ly_list[c]);
                        h_all_lz.push_back(shell.lz_list[c]);
                    }
                    comp_offset += shell.ao_indices.size();
                }

                int *d_ncomponents, *d_comp_offsets, *d_all_lx, *d_all_ly, *d_all_lz;
                cudaMalloc(&d_ncomponents, num_shells * sizeof(int));
                cudaMalloc(&d_comp_offsets, num_shells * sizeof(int));
                cudaMalloc(&d_all_lx, h_all_lx.size() * sizeof(int));
                cudaMalloc(&d_all_ly, h_all_ly.size() * sizeof(int));
                cudaMalloc(&d_all_lz, h_all_lz.size() * sizeof(int));

                cudaMemcpy(d_ncomponents, h_ncomponents.data(), num_shells * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_comp_offsets, h_comp_offsets.data(), num_shells * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_all_lx, h_all_lx.data(), h_all_lx.size() * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_all_ly, h_all_ly.data(), h_all_ly.size() * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_all_lz, h_all_lz.data(), h_all_lz.size() * sizeof(int), cudaMemcpyHostToDevice);

                evaluate_general_shells_kernel<<<grid_size, block_size>>>(
                    d_ao_values, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_ncomponents, d_all_lx, d_all_ly, d_all_lz, d_comp_offsets,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);

                cudaFree(d_ncomponents);
                cudaFree(d_comp_offsets);
                cudaFree(d_all_lx);
                cudaFree(d_all_ly);
                cudaFree(d_all_lz);
            }
            // --- [新增] 停止记录并累加时间 ---
            cudaEventRecord(stop, 0);   // <--- 记录 stop
            cudaEventSynchronize(stop); // <--- 等待 GPU 执行到 stop 点（确保计时准确）

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop); // <--- 计算时间差
            total_kernel_time_ms += milliseconds;             // <--- 累加
                                                              // --------------------------------
            cudaFree(d_ao_start);
            cudaFree(d_atom_indices);
            cudaFree(d_prim_offsets);
            cudaFree(d_prim_counts);
            cudaFree(d_all_exps);
            cudaFree(d_all_coeffs);
        }

        cudaDeviceSynchronize();
        // --- [新增] 输出总时间 & 销毁事件 ---
        std::cout << "Total Pure Kernel Execution Time: " << total_kernel_time_ms << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaMemcpy(out_ao_values, d_ao_values, total_size, cudaMemcpyDeviceToHost);

        cudaFree(d_grid_coords);
        cudaFree(d_atom_coords);
        cudaFree(d_ao_values);
    }
    // fan 1223  aos grids evaluate
    __device__ __forceinline__ double dev_pow_int(double x, int n)
    {
        if (n <= 0)
            return (n == 0 ? 1.0 : 0.0); // n<0 -> 0 (不会被用到，因为前面有系数 l=0)
        double r = 1.0;
        for (int i = 0; i < n; ++i)
            r *= x;
        return r;
    }

    // out_grad layout: [3 x ngrids x nao]
    __global__ void evaluate_single_ao_grad_kernel(
        double *out_grad,          // [3*ngrids*nao]
        const double *grid_coords, // [ngrids*3]
        const double *atom_coord,  // [3]
        const double *exps,        // [nprim]
        const double *coeffs,      // [nprim]
        int nprim,
        int lx, int ly, int lz,
        double fac, // e.g. get_fac(l)
        int ngrids,
        int nao,
        int ao_idx) // which column to write
    {
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        if (g >= ngrids)
            return;

        const double gx = grid_coords[g * 3 + 0];
        const double gy = grid_coords[g * 3 + 1];
        const double gz = grid_coords[g * 3 + 2];

        const double dx = gx - atom_coord[0];
        const double dy = gy - atom_coord[1];
        const double dz = gz - atom_coord[2];
        const double r2 = dx * dx + dy * dy + dz * dz;

        // contracted sums
        double ce = 0.0;   // Σ c * exp(-α r^2)
        double ce_a = 0.0; // Σ c * exp(-α r^2) * α
        for (int p = 0; p < nprim; ++p)
        {
            const double alpha = exps[p];
            const double coeff = coeffs[p];
            const double e = exp(-alpha * r2);
            ce += coeff * e;
            ce_a += coeff * e * alpha;
        }
        ce *= fac;
        const double ce_2a = (-2.0) * fac * ce_a; // Σ c * (-2α) * exp(-α r^2) * fac

        // angular powers
        const double dx_lx = dev_pow_int(dx, lx);
        const double dy_ly = dev_pow_int(dy, ly);
        const double dz_lz = dev_pow_int(dz, lz);
        const double dx_lx_m1 = (lx > 0) ? dev_pow_int(dx, lx - 1) : 0.0;
        const double dy_ly_m1 = (ly > 0) ? dev_pow_int(dy, ly - 1) : 0.0;
        const double dz_lz_m1 = (lz > 0) ? dev_pow_int(dz, lz - 1) : 0.0;

        const double common_yz = dy_ly * dz_lz;
        const double common_xz = dx_lx * dz_lz;
        const double common_xy = dx_lx * dy_ly;

        // CPU 同式：grad_x = dy^ly dz^lz ( lx dx^(lx-1) ce + dx^lx dx ce_2a )
        const double grad_x = common_yz * ((double)lx * dx_lx_m1 * ce + dx_lx * dx * ce_2a);
        const double grad_y = common_xz * ((double)ly * dy_ly_m1 * ce + dy_ly * dy * ce_2a);
        const double grad_z = common_xy * ((double)lz * dz_lz_m1 * ce + dz_lz * dz * ce_2a);

        const int base = g * nao + ao_idx;
        out_grad[0 * (ngrids * nao) + base] = grad_x;
        out_grad[1 * (ngrids * nao) + base] = grad_y;
        out_grad[2 * (ngrids * nao) + base] = grad_z;
    }
    void evaluate_ao_grad_on_grids_gpu_raw(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_grad, // [3 x ngrids x nao]
        int ngrids,
        int nao)
    {
        if (nao != (int)ao_list.size())
            throw std::invalid_argument("nao != ao_list.size()");
        if (ngrids != (int)grid_coords.size())
            throw std::invalid_argument("ngrids != grid_coords.size()");

        std::cout << "Evaluating AO grads: nao=" << nao << ", ngrids=" << ngrids << "\n";

        const size_t total_size = (size_t)3 * (size_t)ngrids * (size_t)nao * sizeof(double);
        std::cout << "Total grad memory requirement: " << total_size / (1024.0 * 1024.0) << " MB\n";

        // device out
        double *d_grad = nullptr;
        cudaMalloc(&d_grad, total_size);
        cudaMemset(d_grad, 0, total_size);

        // flatten grids
        std::vector<double> h_flat_grids((size_t)ngrids * 3);
        for (int i = 0; i < ngrids; ++i)
        {
            h_flat_grids[i * 3 + 0] = grid_coords[i][0];
            h_flat_grids[i * 3 + 1] = grid_coords[i][1];
            h_flat_grids[i * 3 + 2] = grid_coords[i][2];
        }

        double *d_grid_coords = nullptr;
        cudaMalloc(&d_grid_coords, (size_t)ngrids * 3 * sizeof(double));
        cudaMemcpy(d_grid_coords, h_flat_grids.data(),
                   (size_t)ngrids * 3 * sizeof(double), cudaMemcpyHostToDevice);

        const int block_size = 256;
        const int grid_size = (ngrids + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_kernel_time_ms = 0.0f;

        for (int ao_idx = 0; ao_idx < nao; ++ao_idx)
        {
            const AODesc &ao = ao_list[ao_idx];
            const int nprim = (int)ao.exps.size();

            // atom coord (device)
            double atom_coord_h[3] = {
                atom_coords[ao.atom][0],
                atom_coords[ao.atom][1],
                atom_coords[ao.atom][2]};
            double *d_atom_coord = nullptr;
            cudaMalloc(&d_atom_coord, 3 * sizeof(double));
            cudaMemcpy(d_atom_coord, atom_coord_h, 3 * sizeof(double), cudaMemcpyHostToDevice);

            // exps/coeffs (device)
            double *d_exps = nullptr, *d_coeffs = nullptr;
            cudaMalloc(&d_exps, (size_t)nprim * sizeof(double));
            cudaMalloc(&d_coeffs, (size_t)nprim * sizeof(double));
            cudaMemcpy(d_exps, ao.exps.data(), (size_t)nprim * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_coeffs, ao.coeffs.data(), (size_t)nprim * sizeof(double), cudaMemcpyHostToDevice);

            // fac: 用你已有的 get_fac(l)（host 上算好，传进 kernel）
            const double fac = get_fac(ao.l);

            cudaEventRecord(start, 0);
            evaluate_single_ao_grad_kernel<<<grid_size, block_size>>>(
                d_grad,
                d_grid_coords,
                d_atom_coord,
                d_exps,
                d_coeffs,
                nprim,
                ao.lx, ao.ly, ao.lz,
                fac,
                ngrids,
                nao,
                ao_idx);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_kernel_time_ms += ms;

            cudaGetLastError();

            cudaFree(d_atom_coord);
            cudaFree(d_exps);
            cudaFree(d_coeffs);

            if ((ao_idx + 1) % 10 == 0 || ao_idx == nao - 1)
                std::cout << "  Completed " << (ao_idx + 1) << "/" << nao << " AOs\n";
        }

        std::cout << "========================================\n";
        std::cout << "RAW Total PURE Kernel Execution Time: " << total_kernel_time_ms << " ms\n";
        std::cout << "========================================\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(out_grad, d_grad, total_size, cudaMemcpyDeviceToHost);

        cudaFree(d_grid_coords);
        cudaFree(d_grad);

        std::cout << "GPU AO grad evaluation completed!\n";
    }

    // fan 12 23 grouped ao grid
    //  ============================================================
    //  Kernel for s orbitals (l=0)
    //  Output layout: [3*ngrids x nao]
    //  ============================================================
    __global__ void evaluate_s_shells_grad_kernel(
        double *__restrict__ ao_grad, // [3 x ngrids x nao]
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_indices, // size=num_shells, each is the AO index
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        if (g >= ngrids)
            return;

        const double gx = grid_coords[g * 3 + 0];
        const double gy = grid_coords[g * 3 + 1];
        const double gz = grid_coords[g * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            const int ao_idx = shell_ao_indices[s];
            const int atom = shell_atom_indices[s];
            const int p0 = shell_prim_offsets[s];
            const int nprim = shell_prim_counts[s];

            const double ax = atom_coords[atom * 3 + 0];
            const double ay = atom_coords[atom * 3 + 1];
            const double az = atom_coords[atom * 3 + 2];

            const double dx = gx - ax;
            const double dy = gy - ay;
            const double dz = gz - az;
            const double r2 = dx * dx + dy * dy + dz * dz;

            double ce_a = 0.0; // Σ c*e*alpha
            for (int p = 0; p < nprim; ++p)
            {
                const double alpha = all_exps[p0 + p];
                const double coeff = all_coeffs[p0 + p];
                const double e = exp(-alpha * r2);
                ce_a += coeff * e * alpha;
            }
            const double ce_2a = (-2.0) * fac * ce_a;

            const int base = g * nao + ao_idx;
            ao_grad[0 * (ngrids * nao) + base] = ce_2a * dx;
            ao_grad[1 * (ngrids * nao) + base] = ce_2a * dy;
            ao_grad[2 * (ngrids * nao) + base] = ce_2a * dz;
        }
    }
    __global__ void evaluate_p_shells_grad_kernel(
        double *__restrict__ ao_grad,
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start, // px index
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        if (g >= ngrids)
            return;

        const double gx = grid_coords[g * 3 + 0];
        const double gy = grid_coords[g * 3 + 1];
        const double gz = grid_coords[g * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            const int ao0 = shell_ao_start[s];
            const int atom = shell_atom_indices[s];
            const int p0 = shell_prim_offsets[s];
            const int nprim = shell_prim_counts[s];

            const double ax = atom_coords[atom * 3 + 0];
            const double ay = atom_coords[atom * 3 + 1];
            const double az = atom_coords[atom * 3 + 2];

            const double dx = gx - ax;
            const double dy = gy - ay;
            const double dz = gz - az;
            const double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;
            double ce_a = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                const double alpha = all_exps[p0 + p];
                const double coeff = all_coeffs[p0 + p];
                const double e = exp(-alpha * r2);
                ce += coeff * e;
                ce_a += coeff * e * alpha;
            }
            ce *= fac;
            const double ce_2a = (-2.0) * fac * ce_a;

            const int bpx = g * nao + (ao0 + 0);
            const int bpy = g * nao + (ao0 + 1);
            const int bpz = g * nao + (ao0 + 2);

            // px = ce*dx
            ao_grad[0 * (ngrids * nao) + bpx] = ce + ce_2a * dx * dx;
            ao_grad[1 * (ngrids * nao) + bpx] = ce_2a * dy * dx;
            ao_grad[2 * (ngrids * nao) + bpx] = ce_2a * dz * dx;

            // py = ce*dy
            ao_grad[0 * (ngrids * nao) + bpy] = ce_2a * dx * dy;
            ao_grad[1 * (ngrids * nao) + bpy] = ce + ce_2a * dy * dy;
            ao_grad[2 * (ngrids * nao) + bpy] = ce_2a * dz * dy;

            // pz = ce*dz
            ao_grad[0 * (ngrids * nao) + bpz] = ce_2a * dx * dz;
            ao_grad[1 * (ngrids * nao) + bpz] = ce_2a * dy * dz;
            ao_grad[2 * (ngrids * nao) + bpz] = ce + ce_2a * dz * dz;
        }
    }

    // d
    __global__ void evaluate_d_shells_grad_kernel(
        double *__restrict__ ao_grad, // [3 x ngrids x nao]
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start, // base AO index for this d shell
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        if (g >= ngrids)
            return;

        const double gx = grid_coords[g * 3 + 0];
        const double gy = grid_coords[g * 3 + 1];
        const double gz = grid_coords[g * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            const int ao0 = shell_ao_start[s];
            const int atom = shell_atom_indices[s];
            const int p0 = shell_prim_offsets[s];
            const int nprim = shell_prim_counts[s];

            const double ax = atom_coords[atom * 3 + 0];
            const double ay = atom_coords[atom * 3 + 1];
            const double az = atom_coords[atom * 3 + 2];

            const double dx = gx - ax;
            const double dy = gy - ay;
            const double dz = gz - az;
            const double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;   // fac * Σ c e
            double ce_a = 0.0; // fac * Σ c e α  (later multiply -2)
            for (int p = 0; p < nprim; ++p)
            {
                const double alpha = all_exps[p0 + p];
                const double coeff = all_coeffs[p0 + p];
                const double e = exp(-alpha * r2);
                ce += coeff * e;
                ce_a += coeff * e * alpha;
            }
            ce *= fac;
            const double ce_2a = (-2.0) * fac * ce_a;

            const double dx2 = dx * dx;
            const double dy2 = dy * dy;
            const double dz2 = dz * dz;

            // monomials
            const double M_xx = dx2;
            const double M_xy = dx * dy;
            const double M_xz = dx * dz;
            const double M_yy = dy2;
            const double M_yz = dy * dz;
            const double M_zz = dz2;

            // d/dx ce = ce_2a * dx (same for y,z)
            const double dce_dx = ce_2a * dx;
            const double dce_dy = ce_2a * dy;
            const double dce_dz = ce_2a * dz;

            // ∂xM, ∂yM, ∂zM
            // xx
            const double dMxx_dx = 2.0 * dx, dMxx_dy = 0.0, dMxx_dz = 0.0;
            // xy
            const double dMxy_dx = dy, dMxy_dy = dx, dMxy_dz = 0.0;
            // xz
            const double dMxz_dx = dz, dMxz_dy = 0.0, dMxz_dz = dx;
            // yy
            const double dMyy_dx = 0.0, dMyy_dy = 2.0 * dy, dMyy_dz = 0.0;
            // yz
            const double dMyz_dx = 0.0, dMyz_dy = dz, dMyz_dz = dy;
            // zz
            const double dMzz_dx = 0.0, dMzz_dy = 0.0, dMzz_dz = 2.0 * dz;

            // write helper lambda-like macro
            auto write3 = [&](int ao, double dM_dx, double dM_dy, double dM_dz, double M)
            {
                const int base = g * nao + ao;
                ao_grad[0 * (ngrids * nao) + base] = ce * dM_dx + dce_dx * M;
                ao_grad[1 * (ngrids * nao) + base] = ce * dM_dy + dce_dy * M;
                ao_grad[2 * (ngrids * nao) + base] = ce * dM_dz + dce_dz * M;
            };

            // order: xx, xy, xz, yy, yz, zz
            write3(ao0 + 0, dMxx_dx, dMxx_dy, dMxx_dz, M_xx);
            write3(ao0 + 1, dMxy_dx, dMxy_dy, dMxy_dz, M_xy);
            write3(ao0 + 2, dMxz_dx, dMxz_dy, dMxz_dz, M_xz);
            write3(ao0 + 3, dMyy_dx, dMyy_dy, dMyy_dz, M_yy);
            write3(ao0 + 4, dMyz_dx, dMyz_dy, dMyz_dz, M_yz);
            write3(ao0 + 5, dMzz_dx, dMzz_dy, dMzz_dz, M_zz);
        }
    }
    // f
    __global__ void evaluate_f_shells_grad_kernel(
        double *__restrict__ ao_grad, // [3 x ngrids x nao]
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start, // base AO index for this f shell
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        if (g >= ngrids)
            return;

        const double gx = grid_coords[g * 3 + 0];
        const double gy = grid_coords[g * 3 + 1];
        const double gz = grid_coords[g * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            const int ao0 = shell_ao_start[s];
            const int atom = shell_atom_indices[s];
            const int p0 = shell_prim_offsets[s];
            const int nprim = shell_prim_counts[s];

            const double ax = atom_coords[atom * 3 + 0];
            const double ay = atom_coords[atom * 3 + 1];
            const double az = atom_coords[atom * 3 + 2];

            const double dx = gx - ax;
            const double dy = gy - ay;
            const double dz = gz - az;
            const double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;
            double ce_a = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                const double alpha = all_exps[p0 + p];
                const double coeff = all_coeffs[p0 + p];
                const double e = exp(-alpha * r2);
                ce += coeff * e;
                ce_a += coeff * e * alpha;
            }
            ce *= fac;
            const double ce_2a = (-2.0) * fac * ce_a;

            const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
            const double dx3 = dx2 * dx, dy3 = dy2 * dy, dz3 = dz2 * dz;

            const double dce_dx = ce_2a * dx;
            const double dce_dy = ce_2a * dy;
            const double dce_dz = ce_2a * dz;

            // monomials (order as your value kernel)
            const double M_xxx = dx3;
            const double M_xxy = dx2 * dy;
            const double M_xxz = dx2 * dz;
            const double M_xyy = dx * dy2;
            const double M_xyz = dx * dy * dz;
            const double M_xzz = dx * dz2;
            const double M_yyy = dy3;
            const double M_yyz = dy2 * dz;
            const double M_yzz = dy * dz2;
            const double M_zzz = dz3;

            // derivatives of monomials
            // xxx
            const double dMxxx_dx = 3.0 * dx2, dMxxx_dy = 0.0, dMxxx_dz = 0.0;
            // xxy
            const double dMxxy_dx = 2.0 * dx * dy, dMxxy_dy = dx2, dMxxy_dz = 0.0;
            // xxz
            const double dMxxz_dx = 2.0 * dx * dz, dMxxz_dy = 0.0, dMxxz_dz = dx2;
            // xyy
            const double dMxyy_dx = dy2, dMxyy_dy = 2.0 * dx * dy, dMxyy_dz = 0.0;
            // xyz
            const double dMxyz_dx = dy * dz, dMxyz_dy = dx * dz, dMxyz_dz = dx * dy;
            // xzz
            const double dMxzz_dx = dz2, dMxzz_dy = 0.0, dMxzz_dz = 2.0 * dx * dz;
            // yyy
            const double dMyyy_dx = 0.0, dMyyy_dy = 3.0 * dy2, dMyyy_dz = 0.0;
            // yyz
            const double dMyyz_dx = 0.0, dMyyz_dy = 2.0 * dy * dz, dMyyz_dz = dy2;
            // yzz
            const double dMyzz_dx = 0.0, dMyzz_dy = dz2, dMyzz_dz = 2.0 * dy * dz;
            // zzz
            const double dMzzz_dx = 0.0, dMzzz_dy = 0.0, dMzzz_dz = 3.0 * dz2;

            auto write3 = [&](int ao, double dM_dx, double dM_dy, double dM_dz, double M)
            {
                const int base = g * nao + ao;
                ao_grad[0 * (ngrids * nao) + base] = ce * dM_dx + dce_dx * M;
                ao_grad[1 * (ngrids * nao) + base] = ce * dM_dy + dce_dy * M;
                ao_grad[2 * (ngrids * nao) + base] = ce * dM_dz + dce_dz * M;
            };

            write3(ao0 + 0, dMxxx_dx, dMxxx_dy, dMxxx_dz, M_xxx);
            write3(ao0 + 1, dMxxy_dx, dMxxy_dy, dMxxy_dz, M_xxy);
            write3(ao0 + 2, dMxxz_dx, dMxxz_dy, dMxxz_dz, M_xxz);
            write3(ao0 + 3, dMxyy_dx, dMxyy_dy, dMxyy_dz, M_xyy);
            write3(ao0 + 4, dMxyz_dx, dMxyz_dy, dMxyz_dz, M_xyz);
            write3(ao0 + 5, dMxzz_dx, dMxzz_dy, dMxzz_dz, M_xzz);
            write3(ao0 + 6, dMyyy_dx, dMyyy_dy, dMyyy_dz, M_yyy);
            write3(ao0 + 7, dMyyz_dx, dMyyz_dy, dMyyz_dz, M_yyz);
            write3(ao0 + 8, dMyzz_dx, dMyzz_dy, dMyzz_dz, M_yzz);
            write3(ao0 + 9, dMzzz_dx, dMzzz_dy, dMzzz_dz, M_zzz);
        }
    }

    // gener

    __global__ void evaluate_general_shells_grad_kernel(
        double *__restrict__ ao_grad, // [3 x ngrids x nao]
        const double *__restrict__ grid_coords,
        const double *__restrict__ atom_coords,
        const int *__restrict__ shell_ao_start,
        const int *__restrict__ shell_atom_indices,
        const int *__restrict__ shell_prim_offsets,
        const int *__restrict__ shell_prim_counts,
        const int *__restrict__ shell_ncomponents,
        const int *__restrict__ all_lx,
        const int *__restrict__ all_ly,
        const int *__restrict__ all_lz,
        const int *__restrict__ ao_component_offsets,
        const double *__restrict__ all_exps,
        const double *__restrict__ all_coeffs,
        double fac,
        int num_shells,
        int ngrids,
        int nao)
    {
        int g = blockIdx.x * blockDim.x + threadIdx.x;
        if (g >= ngrids)
            return;

        const double gx = grid_coords[g * 3 + 0];
        const double gy = grid_coords[g * 3 + 1];
        const double gz = grid_coords[g * 3 + 2];

        for (int s = 0; s < num_shells; ++s)
        {
            const int ao0 = shell_ao_start[s];
            const int atom = shell_atom_indices[s];
            const int p0 = shell_prim_offsets[s];
            const int nprim = shell_prim_counts[s];
            const int ncomp = shell_ncomponents[s];
            const int coff = ao_component_offsets[s];

            const double ax = atom_coords[atom * 3 + 0];
            const double ay = atom_coords[atom * 3 + 1];
            const double az = atom_coords[atom * 3 + 2];

            const double dx = gx - ax;
            const double dy = gy - ay;
            const double dz = gz - az;
            const double r2 = dx * dx + dy * dy + dz * dz;

            double ce = 0.0;
            double ce_a = 0.0;
            for (int p = 0; p < nprim; ++p)
            {
                const double alpha = all_exps[p0 + p];
                const double coeff = all_coeffs[p0 + p];
                const double e = exp(-alpha * r2);
                ce += coeff * e;
                ce_a += coeff * e * alpha;
            }
            ce *= fac;
            const double ce_2a = (-2.0) * fac * ce_a;

            for (int c = 0; c < ncomp; ++c)
            {
                const int lx = all_lx[coff + c];
                const int ly = all_ly[coff + c];
                const int lz = all_lz[coff + c];

                const double dx_lx = dev_power(dx, lx);
                const double dy_ly = dev_power(dy, ly);
                const double dz_lz = dev_power(dz, lz);
                const double dx_lx_m1 = (lx > 0) ? dev_power(dx, lx - 1) : 0.0;
                const double dy_ly_m1 = (ly > 0) ? dev_power(dy, ly - 1) : 0.0;
                const double dz_lz_m1 = (lz > 0) ? dev_power(dz, lz - 1) : 0.0;

                const double common_yz = dy_ly * dz_lz;
                const double common_xz = dx_lx * dz_lz;
                const double common_xy = dx_lx * dy_ly;

                const double gxv = common_yz * ((double)lx * dx_lx_m1 * ce + dx_lx * dx * ce_2a);
                const double gyv = common_xz * ((double)ly * dy_ly_m1 * ce + dy_ly * dy * ce_2a);
                const double gzv = common_xy * ((double)lz * dz_lz_m1 * ce + dz_lz * dz * ce_2a);

                const int base = g * nao + (ao0 + c);
                ao_grad[0 * (ngrids * nao) + base] = gxv;
                ao_grad[1 * (ngrids * nao) + base] = gyv;
                ao_grad[2 * (ngrids * nao) + base] = gzv;
            }
        }
    }

    // host kernel
    void evaluate_aos_grad_gpu_shell_grouped(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_grad, // [3 x ngrids x nao]
        int ngrids,
        int nao)
    {
        auto shell_groups = group_aos_by_shell(ao_list);

        // device out
        const size_t total_size = (size_t)3 * (size_t)ngrids * (size_t)nao * sizeof(double);
        double *d_ao_grad = nullptr;
        cudaMalloc(&d_ao_grad, total_size);
        cudaMemset(d_ao_grad, 0, total_size);

        // grid coords [ngrids x 3]
        std::vector<double> h_flat_grids((size_t)ngrids * 3);
        for (int i = 0; i < ngrids; ++i)
        {
            h_flat_grids[i * 3 + 0] = grid_coords[i][0];
            h_flat_grids[i * 3 + 1] = grid_coords[i][1];
            h_flat_grids[i * 3 + 2] = grid_coords[i][2];
        }
        double *d_grid_coords = nullptr;
        cudaMalloc(&d_grid_coords, (size_t)ngrids * 3 * sizeof(double));
        cudaMemcpy(d_grid_coords, h_flat_grids.data(),
                   (size_t)ngrids * 3 * sizeof(double), cudaMemcpyHostToDevice);

        // atom coords [natoms x 3]
        const int natoms = (int)atom_coords.size();
        std::vector<double> h_atom_coords((size_t)natoms * 3);
        for (int i = 0; i < natoms; ++i)
        {
            h_atom_coords[i * 3 + 0] = atom_coords[i][0];
            h_atom_coords[i * 3 + 1] = atom_coords[i][1];
            h_atom_coords[i * 3 + 2] = atom_coords[i][2];
        }
        double *d_atom_coords = nullptr;
        cudaMalloc(&d_atom_coords, (size_t)natoms * 3 * sizeof(double));
        cudaMemcpy(d_atom_coords, h_atom_coords.data(),
                   (size_t)natoms * 3 * sizeof(double), cudaMemcpyHostToDevice);

        const int block_size = 256;
        const int grid_size = (ngrids + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float total_kernel_time_ms = 0.0f;

        for (const auto &[l, shells] : shell_groups)
        {
            const int num_shells = (int)shells.size();
            const double fac = get_fac(l);

            // per-shell arrays
            std::vector<int> h_ao_start, h_atom_indices, h_prim_offsets, h_prim_counts;
            std::vector<double> h_all_exps, h_all_coeffs;

            h_ao_start.reserve(num_shells);
            h_atom_indices.reserve(num_shells);
            h_prim_offsets.reserve(num_shells);
            h_prim_counts.reserve(num_shells);

            int prim_offset = 0;
            for (const auto &shell : shells)
            {
                // s: 单 AO index；p/d/...: 起始 index
                h_ao_start.push_back(shell.ao_indices[0]);
                h_atom_indices.push_back(shell.atom_idx);
                h_prim_offsets.push_back(prim_offset);
                h_prim_counts.push_back(shell.nprim);

                for (double e : shell.exps)
                    h_all_exps.push_back(e);
                for (double c : shell.coeffs)
                    h_all_coeffs.push_back(c);
                prim_offset += shell.nprim;
            }

            int *d_ao_start = nullptr, *d_atom_indices = nullptr, *d_prim_offsets = nullptr, *d_prim_counts = nullptr;
            double *d_all_exps = nullptr, *d_all_coeffs = nullptr;

            cudaMalloc(&d_ao_start, (size_t)num_shells * sizeof(int));
            cudaMalloc(&d_atom_indices, (size_t)num_shells * sizeof(int));
            cudaMalloc(&d_prim_offsets, (size_t)num_shells * sizeof(int));
            cudaMalloc(&d_prim_counts, (size_t)num_shells * sizeof(int));
            cudaMalloc(&d_all_exps, (size_t)h_all_exps.size() * sizeof(double));
            cudaMalloc(&d_all_coeffs, (size_t)h_all_coeffs.size() * sizeof(double));

            cudaMemcpy(d_ao_start, h_ao_start.data(), (size_t)num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_atom_indices, h_atom_indices.data(), (size_t)num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prim_offsets, h_prim_offsets.data(), (size_t)num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prim_counts, h_prim_counts.data(), (size_t)num_shells * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_exps, h_all_exps.data(), (size_t)h_all_exps.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_all_coeffs, h_all_coeffs.data(), (size_t)h_all_coeffs.size() * sizeof(double), cudaMemcpyHostToDevice);

            cudaEventRecord(start, 0);

            if (l == 0)
            {
                // 注意：s kernel 参数名字叫 shell_ao_indices（单 index）
                gansu::dft::chemgrid::evaluate_s_shells_grad_kernel<<<grid_size, block_size>>>(
                    d_ao_grad, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else if (l == 1)
            {
                gansu::dft::chemgrid::evaluate_p_shells_grad_kernel<<<grid_size, block_size>>>(
                    d_ao_grad, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else if (l == 2)
            {
                evaluate_d_shells_grad_kernel<<<grid_size, block_size>>>(
                    d_ao_grad, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }
            else if (l == 3)
            {
                evaluate_f_shells_grad_kernel<<<grid_size, block_size>>>(
                    d_ao_grad, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);
            }

            else
            {
                // 统一走 general：需要组件表
                std::vector<int> h_ncomponents, h_comp_offsets;
                std::vector<int> h_all_lx, h_all_ly, h_all_lz;

                h_ncomponents.reserve(num_shells);
                h_comp_offsets.reserve(num_shells);

                int comp_offset = 0;
                for (const auto &shell : shells)
                {
                    const int ncomp = (int)shell.ao_indices.size();
                    h_ncomponents.push_back(ncomp);
                    h_comp_offsets.push_back(comp_offset);
                    for (int c = 0; c < ncomp; ++c)
                    {
                        h_all_lx.push_back(shell.lx_list[c]);
                        h_all_ly.push_back(shell.ly_list[c]);
                        h_all_lz.push_back(shell.lz_list[c]);
                    }
                    comp_offset += ncomp;
                }

                int *d_ncomponents = nullptr, *d_comp_offsets = nullptr, *d_all_lx = nullptr, *d_all_ly = nullptr, *d_all_lz = nullptr;
                cudaMalloc(&d_ncomponents, (size_t)num_shells * sizeof(int));
                cudaMalloc(&d_comp_offsets, (size_t)num_shells * sizeof(int));
                cudaMalloc(&d_all_lx, (size_t)h_all_lx.size() * sizeof(int));
                cudaMalloc(&d_all_ly, (size_t)h_all_ly.size() * sizeof(int));
                cudaMalloc(&d_all_lz, (size_t)h_all_lz.size() * sizeof(int));

                cudaMemcpy(d_ncomponents, h_ncomponents.data(), (size_t)num_shells * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_comp_offsets, h_comp_offsets.data(), (size_t)num_shells * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_all_lx, h_all_lx.data(), (size_t)h_all_lx.size() * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_all_ly, h_all_ly.data(), (size_t)h_all_ly.size() * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_all_lz, h_all_lz.data(), (size_t)h_all_lz.size() * sizeof(int), cudaMemcpyHostToDevice);

                gansu::dft::chemgrid::evaluate_general_shells_grad_kernel<<<grid_size, block_size>>>(
                    d_ao_grad, d_grid_coords, d_atom_coords,
                    d_ao_start, d_atom_indices, d_prim_offsets, d_prim_counts,
                    d_ncomponents, d_all_lx, d_all_ly, d_all_lz, d_comp_offsets,
                    d_all_exps, d_all_coeffs,
                    fac, num_shells, ngrids, nao);

                cudaFree(d_ncomponents);
                cudaFree(d_comp_offsets);
                cudaFree(d_all_lx);
                cudaFree(d_all_ly);
                cudaFree(d_all_lz);
            }

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_kernel_time_ms += ms;

            cudaGetLastError();

            cudaFree(d_ao_start);
            cudaFree(d_atom_indices);
            cudaFree(d_prim_offsets);
            cudaFree(d_prim_counts);
            cudaFree(d_all_exps);
            cudaFree(d_all_coeffs);
        }

        cudaDeviceSynchronize();
        std::cout << "Total Pure Kernel Execution Time (grad): " << total_kernel_time_ms << " ms\n";
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(out_grad, d_ao_grad, total_size, cudaMemcpyDeviceToHost);

        cudaFree(d_grid_coords);
        cudaFree(d_atom_coords);
        cudaFree(d_ao_grad);
    }

}