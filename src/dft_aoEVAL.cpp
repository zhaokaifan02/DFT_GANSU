#include "dft_aoEVAL.hpp"
#include <stdexcept>
#include <cstring>

namespace chemgrid::AOEval
{
    static inline void CUDA_CHECK_THROW(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));
    }

    void AODeviceBuffers::free_all()
    {
        auto f = [](void *p)
        { if (p) cudaFree(p); };
        f(d_center);
        f(d_lx);
        f(d_ly);
        f(d_lz);
        f(d_off);
        f(d_npr);
        f(d_ax);
        f(d_ay);
        f(d_az);
        f(d_exps);
        f(d_coeffs);
        f(d_gx);
        f(d_gy);
        f(d_gz);
        f(d_ao);
        d_center = d_lx = d_ly = d_lz = d_off = d_npr = nullptr;
        d_ax = d_ay = d_az = nullptr;
        d_exps = d_coeffs = nullptr;
        d_gx = d_gy = d_gz = nullptr;
        d_ao = nullptr;
    }

    AOFlatHost flatten_to_soa(const std::vector<AODesc> &aos,
                              const std::vector<std::array<double, 3>> &atm_coords)
    {
        AOFlatHost H;
        H.nao = (int)aos.size();
        H.natm = (int)atm_coords.size();

        H.center_of_ao.resize(H.nao);
        H.lx.resize(H.nao);
        H.ly.resize(H.nao);
        H.lz.resize(H.nao);
        H.prim_offset.resize(H.nao);
        H.nprim.resize(H.nao);

        H.ax.resize(H.natm);
        H.ay.resize(H.natm);
        H.az.resize(H.natm);
        for (int A = 0; A < H.natm; ++A)
        {
            H.ax[A] = atm_coords[A][0];
            H.ay[A] = atm_coords[A][1];
            H.az[A] = atm_coords[A][2];
        }

        // primitives 池拼接
        int off = 0;
        for (int mu = 0; mu < H.nao; ++mu)
        {
            const AODesc &ao = aos[mu];
            const int npr = (int)ao.exps.size();
            if ((int)ao.coeffs.size() != npr)
                throw std::runtime_error("AODesc exps/coeffs size mismatch");
            H.center_of_ao[mu] = ao.atom;
            H.lx[mu] = ao.lx;
            H.ly[mu] = ao.ly;
            H.lz[mu] = ao.lz;
            H.prim_offset[mu] = off;
            H.nprim[mu] = npr;

            H.exps_pool.insert(H.exps_pool.end(), ao.exps.begin(), ao.exps.end());
            H.coeffs_pool.insert(H.coeffs_pool.end(), ao.coeffs.begin(), ao.coeffs.end());
            off += npr;
        }
        H.nprim_tot = off;
        return H;
    }

    void upload_AO_to_device(const AOFlatHost &A, AODevice &AOd, AODeviceBuffers &buf)
    {
        // alloc & copy meta
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_center, sizeof(int) * A.nao));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_lx, sizeof(int) * A.nao));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_ly, sizeof(int) * A.nao));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_lz, sizeof(int) * A.nao));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_off, sizeof(int) * A.nao));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_npr, sizeof(int) * A.nao));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_center, A.center_of_ao.data(), sizeof(int) * A.nao, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_lx, A.lx.data(), sizeof(int) * A.nao, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_ly, A.ly.data(), sizeof(int) * A.nao, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_lz, A.lz.data(), sizeof(int) * A.nao, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_off, A.prim_offset.data(), sizeof(int) * A.nao, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_npr, A.nprim.data(), sizeof(int) * A.nao, cudaMemcpyHostToDevice));

        // atoms
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_ax, sizeof(double) * A.natm));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_ay, sizeof(double) * A.natm));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_az, sizeof(double) * A.natm));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_ax, A.ax.data(), sizeof(double) * A.natm, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_ay, A.ay.data(), sizeof(double) * A.natm, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_az, A.az.data(), sizeof(double) * A.natm, cudaMemcpyHostToDevice));

        // pools
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_exps, sizeof(double) * A.nprim_tot));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_coeffs, sizeof(double) * A.nprim_tot));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_exps, A.exps_pool.data(), sizeof(double) * A.nprim_tot, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_coeffs, A.coeffs_pool.data(), sizeof(double) * A.nprim_tot, cudaMemcpyHostToDevice));

        // fill AOd view
        AOd.nao = A.nao;
        AOd.natm = A.natm;
        AOd.nprim_tot = A.nprim_tot;
        AOd.center_of_ao = buf.d_center;
        AOd.lx = buf.d_lx;
        AOd.ly = buf.d_ly;
        AOd.lz = buf.d_lz;
        AOd.prim_offset = buf.d_off;
        AOd.nprim = buf.d_npr;
        AOd.ax = buf.d_ax;
        AOd.ay = buf.d_ay;
        AOd.az = buf.d_az;
        AOd.exps_pool = buf.d_exps;
        AOd.coeffs_pool = buf.d_coeffs;
    }

    void upload_grids_to_device(const std::vector<std::array<double, 3>> &grids,
                                AODeviceBuffers &buf)
    {
        const int G = (int)grids.size();
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_gx, sizeof(double) * G));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_gy, sizeof(double) * G));
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_gz, sizeof(double) * G));

        std::vector<double> gx(G), gy(G), gz(G);
        for (int g = 0; g < G; ++g)
        {
            gx[g] = grids[g][0];
            gy[g] = grids[g][1];
            gz[g] = grids[g][2];
        }
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_gx, gx.data(), sizeof(double) * G, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_gy, gy.data(), sizeof(double) * G, cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(buf.d_gz, gz.data(), sizeof(double) * G, cudaMemcpyHostToDevice));
    }

    void download_ao(std::vector<double> &ao_host, int nao, int G, const AODeviceBuffers &buf)
    {
        ao_host.resize((size_t)nao * G);
        CUDA_CHECK_THROW(cudaMemcpy(ao_host.data(), buf.d_ao, sizeof(double) * (size_t)nao * G, cudaMemcpyDeviceToHost));
    }

    // 一站式：Host 入，GPU 算，Host 出
    void evaluate_AO_on_grid_GPU_AOmajor(
        const std::vector<AODesc> &aos,
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::vector<std::array<double, 3>> &grids,
        std::vector<double> &ao_out)
    {
        AOFlatHost H = flatten_to_soa(aos, atm_coords);
        AODeviceBuffers buf;
        AODevice AOd;

        upload_AO_to_device(H, AOd, buf);
        upload_grids_to_device(grids, buf);

        const int G = (int)grids.size();
        CUDA_CHECK_THROW(cudaMalloc(&buf.d_ao, sizeof(double) * (size_t)H.nao * G));

        CUDA_CHECK_THROW(launch_eval_AO_on_grid_AOmajor(AOd, buf.d_gx, buf.d_gy, buf.d_gz, G, buf.d_ao));
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        download_ao(ao_out, H.nao, G, buf);
        buf.free_all();
    }

}