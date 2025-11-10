/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */


#include "gpu_kernels.hpp"
#include "utils.hpp"

namespace gansu::gpu{


/**
 * @brief CUDA kernel for inverse of square root for individual values of input vectors
 * @param d_eigenvalues Device pointer storing the eigenvalues as a vector
 * @param size Size of the input vector
 * @details This function computes the inverse of the square root of each element of the input vector.
 *         The input vector is modified in place.
 */
 __global__ void inverseSqrt_kernel(real_t* d_eigenvalues, const size_t size, const double threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double value = d_eigenvalues[idx];
        if (value < threshold) {
            d_eigenvalues[idx] = 0.0; // Avoid division by zero
        }else{
            d_eigenvalues[idx] = 1.0 / __dsqrt_rn(value);
        }
    }
}




/**
 * @brief CUDA kernel for computing the density matrix for restricted Hartree-Fock
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix Device pointer to the density matrix, each of orbital elements has exactly 2 electrons
 * @param num_electron Number of electrons, must be even
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 * @details The density matrix is given by \f$ D_{ij} = 2 \sum_{k=1}^{N/2} C_{ik} C_{jk} \f$.
 */
 __global__ void computeDensityMatrix_RHF_kernel(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum = 0.0;
    for (size_t k = 0; k < num_electron / 2; k++) {
        sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    d_density_matrix[id] = 2.0 * sum;
}

/**
 * @brief CUDA kernel for computing the density matrix for unrestricted Hartree-Fock
 * @param d_coefficient_matrix Device pointer to the coefficient matrix (alpha or beta)
 * @param d_density_matrix Device pointer to the density matrix (alpha or beta)
 * @param num_spin Number of electrons, must be number of electrons for the alpha or beta spin
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 * @details The density matrix is given by \f$ D_{ij} = \sum_{k=1}^{N} C_{ik} C_{jk} \f$.
 */
 __global__ void computeDensityMatrix_UHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix, const int num_spin, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum = 0.0;
    for (size_t k = 0; k < num_spin; k++) {
        sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    d_density_matrix[id] = sum;
}



/**
 * @brief CUDA kernel for computing the density matrix for ROHF
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix_closed Device pointer to the density matrix (closed-shell)
 * @param d_density_matrix_oepn Device pointer to the density matrix (open-shell)
 * @param d_density_matrix Device pointer to the density matrix (sum of closed-shell and open-shell)
 * @param num_closed Number of closed-shell orbitals
 * @param num_open Number of open-shell orbitals
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 */
 __global__ void computeDensityMatrix_ROHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix_closed, double* d_density_matrix_open, double* d_density_matrix, const int num_closed, const int num_open, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum_closed = 0.0;
    for (size_t k = 0; k < num_closed; k++) {
        sum_closed += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    sum_closed *= 2.0; // closedd shell (2 electrons per orbital)
    d_density_matrix_closed[id] = sum_closed; 

    real_t sum_open = 0.0;
    for (size_t k = num_closed; k < num_closed+num_open; k++) {
        sum_open += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    sum_open *= 1.0; // open shell (1 electron per orbital)
    d_density_matrix_open[id] = sum_open;

    d_density_matrix[id] = sum_closed + sum_open;
}



/**
 * @brief transposeMatrixInPlace_kernel CUDA kernel for transposing a matrix in place
 * @param d_matrix Device pointer to the matrix
 * @param size Size of the matrix
 */
__global__ void transposeMatrixInPlace_kernel(real_t* d_matrix, int size)
{
    if (blockIdx.x < blockIdx.y) {
        return;
    }
    const int xid = blockDim.x * blockIdx.x + threadIdx.x;
    const int yid = blockDim.y * blockIdx.y + threadIdx.y;
    if (xid < yid || xid >= size || yid >= size) {
        return;
    }

    //__shared__ real_t s_src[WARP_SIZE][WARP_SIZE];
    //__shared__ real_t s_dst[WARP_SIZE][WARP_SIZE];
    __shared__ real_t s_src[WARP_SIZE][WARP_SIZE + 1];
    __shared__ real_t s_dst[WARP_SIZE][WARP_SIZE + 1];
    s_src[threadIdx.y][threadIdx.x] = d_matrix[size * yid + xid];
    s_dst[threadIdx.y][threadIdx.x] = d_matrix[size * xid + yid];

    __syncthreads();

    d_matrix[size * yid + xid] = s_dst[threadIdx.y][threadIdx.x];
    d_matrix[size * xid + yid] = s_src[threadIdx.y][threadIdx.x];
}

/**
 * @brief CUDA kernel for computing weight sum matices sum(W[i] * B[i]).
 *
 * @param d_J Output result matrix (MxM) in device memory.
 * @param d_B Input matrices (N matrices of size MxM).
 * @param d_W Scalars (size N).
 * @param M Dimension of matrices (M x M).
 * @param N Number of matrices.
 * @param accumulated If true, the result is accumulated to the output matrix.
 */
__global__ void weighted_sum_matrices_kernel(double* d_J, const double* d_B, const double* d_W, const int M, const int N, const bool accumulated) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= M * M) return;

    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
        sum += d_W[j] * d_B[j * M * M + id];  // Apply scalar multiplication and accumulate
    }

    if(accumulated){
        d_J[id] += sum;
    }else{
        d_J[id] = sum;
    }
}


/**
 * @brief CUDA kernel for computing sum matices sum(B[i]).
 *
 * @param d_J Output result matrix (MxM) in device memory.
 * @param d_B Input matrices (N matrices of size MxM).
 * @param M Dimension of matrices (M x M).
 * @param N Number of matrices.
 * @param accumulated If true, the result is accumulated to the output matrix.
 */
__global__ void sum_matrices_kernel(double* d_K, const double* d_B, const int M, const int N, const bool accumulated) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= M * M) return;

    double sum = 0.0;
    for (int p = 0; p < N; p++) {
        sum += d_B[p * M * M + id];  // Apply scalar multiplication and accumulate
    }

    if(accumulated){
        d_K[id] += sum;
    }else{
        d_K[id] = sum;
    }
}


__global__ void computeFockMatrix_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix, int num_basis)
{
    const int bra = blockIdx.x;
    const int i = bra / num_basis;
    const int j = bra % num_basis;

    // 2-fold symmetry (vertical)
    /*
    const short j = __real_t2int_rn((__dsqrt_rn(8 * bra + 1) - 1) / 2);
    const short i = bra - j * (j + 1) / 2;
    const int uid = num_basis * i + j;
    const int lid = num_basis * j + i;
    */

    const size_t l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ real_t s_F_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_F_ij[0] = 0.0;
    }
    __syncthreads();

    real_t sum = 0.0;
    size_t eid1, eid2;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            eid1 = get_1d_indexM4(i, j, k, l, num_basis);
            //eid2 = get_1d_indexM4(i, l, k, j, num_basis);
            eid2 = get_1d_indexM4(i, k, j, l, num_basis);
            sum += (d_eri[eid1] - 0.5 * d_eri[eid2]) * d_density_matrix[num_basis * k + l];

            // sum += (2.0*d_eri[eid1] -  d_eri[eid2]) * d_density_matrix[num_basis * k + l];
            // sum += (d_eri[eid1] ) * d_density_matrix[num_basis * k + l];
            // sum += (- 0.5 * d_eri[eid2]) * d_density_matrix[num_basis * k + l];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_F_ij, sum);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_fock_matrix[bra] = s_F_ij[0] + d_core_hamiltonian_matrix[bra];
        //g_fock[uid] = g_fock[lid] = s_F_ij[0] + d_core_hamiltonian_matrix[uid];   // 2-fold symmetry
        //g_fock[bra] = s_F_ij[0];  // use cuBLAS
    }
}


__global__ void computeFockMatrix_UHF_kernel(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, int num_basis)
{
    const int bra = blockIdx.x;
    const int i = bra / num_basis;
    const int j = bra % num_basis;

    // 2-fold symmetry (vertical)
    /*
    const short j = __real_t2int_rn((__dsqrt_rn(8 * bra + 1) - 1) / 2);
    const short i = bra - j * (j + 1) / 2;
    const int uid = num_basis * i + j;
    const int lid = num_basis * j + i;
    */

    const size_t l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ real_t s_Fa_ij[1];
    __shared__ real_t s_Fb_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_Fa_ij[0] = 0.0;
        s_Fb_ij[0] = 0.0;
    }
    __syncthreads();

    real_t sum_a = 0.0;
    real_t sum_b = 0.0;
    size_t eid1, eid2;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            eid1 = get_1d_indexM4(i, j, k, l, num_basis);
            //eid2 = get_1d_indexM4(i, l, k, j, num_basis);
            eid2 = get_1d_indexM4(i, k, j, l, num_basis);
            sum_a += (d_density_matrix_a[num_basis * k + l]+d_density_matrix_b[num_basis * k + l]) * d_eri[eid1] - d_density_matrix_a[num_basis * k + l] * d_eri[eid2];
            sum_b += (d_density_matrix_a[num_basis * k + l]+d_density_matrix_b[num_basis * k + l]) * d_eri[eid1] - d_density_matrix_b[num_basis * k + l] * d_eri[eid2];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum_a += __shfl_down_sync(FULL_MASK, sum_a, offset);
        sum_b += __shfl_down_sync(FULL_MASK, sum_b, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_Fa_ij, sum_a);
        atomicAdd(s_Fb_ij, sum_b);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_fock_matrix_a[bra] = s_Fa_ij[0] + d_core_hamiltonian_matrix[bra];
        d_fock_matrix_b[bra] = s_Fb_ij[0] + d_core_hamiltonian_matrix[bra];
        //g_fock[uid] = g_fock[lid] = s_F_ij[0] + d_core_hamiltonian_matrix[uid];   // 2-fold symmetry
        //g_fock[bra] = s_F_ij[0];  // use cuBLAS
    }
}



__global__ void computeFockMatrix_ROHF_kernel(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, int num_basis)
{
    const int bra = blockIdx.x;
    const int i = bra / num_basis;
    const int j = bra % num_basis;

    // 2-fold symmetry (vertical)
    /*
    const short j = __real_t2int_rn((__dsqrt_rn(8 * bra + 1) - 1) / 2);
    const short i = bra - j * (j + 1) / 2;
    const int uid = num_basis * i + j;
    const int lid = num_basis * j + i;
    */

    const size_t l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ real_t s_J_closed_ij[1];
    __shared__ real_t s_J_open_ij[1];
    __shared__ real_t s_K_closed_ij[1];
    __shared__ real_t s_K_open_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_J_closed_ij[0] = 0.0;
        s_J_open_ij[0] = 0.0;
        s_K_closed_ij[0] = 0.0;
        s_K_open_ij[0] = 0.0;
    }
    __syncthreads();

    real_t J_closed = 0.0;
    real_t J_open = 0.0;
    real_t K_closed = 0.0;
    real_t K_open = 0.0;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            const real_t eri_ijkl = d_eri[get_1d_indexM4(i, j, k, l, num_basis)];
            const real_t eri_ikjl = d_eri[get_1d_indexM4(i, k, j, l, num_basis)];
            J_closed += d_density_matrix_closed[num_basis * k + l] * eri_ijkl;
            J_open   += d_density_matrix_open  [num_basis * k + l] * eri_ijkl;
            K_closed += d_density_matrix_closed[num_basis * k + l] * eri_ikjl;
            K_open   += d_density_matrix_open  [num_basis * k + l] * eri_ikjl;
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        J_closed += __shfl_down_sync(FULL_MASK, J_closed, offset);
        J_open  += __shfl_down_sync(FULL_MASK, J_open,  offset);
        K_closed += __shfl_down_sync(FULL_MASK, K_closed, offset);
        K_open  += __shfl_down_sync(FULL_MASK, K_open,  offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_J_closed_ij, J_closed);
        atomicAdd(s_J_open_ij, J_open);
        atomicAdd(s_K_closed_ij, K_closed);
        atomicAdd(s_K_open_ij, K_open);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_fock_matrix_closed[bra] = d_core_hamiltonian_matrix[bra] + s_J_closed_ij[0] - 0.5 * s_K_closed_ij[0] + s_J_open_ij[0] - 0.5 * s_K_open_ij[0];
        d_fock_matrix_open[bra]  = 0.5 * (d_core_hamiltonian_matrix[bra] + s_J_closed_ij[0] - 0.5 * s_K_closed_ij[0] + s_J_open_ij[0] - s_K_open_ij[0]);
    }
}


__global__ void computeUnifiedFockMatrix_ROHF_kernel(const real_t* d_fock_mo_closed_matrix, const real_t* d_fock_mo_open_matrix, const ROHF_ParameterSet rohf_params, real_t* d_unified_fock_matrix, const int num_closed, const int num_open, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * (num_basis+1) / 2) return;

    const size_t2 ij = index1to2(id, true);
    size_t i,j;
    if(ij.x < ij.y){
        i = ij.x;
        j = ij.y;
    }else{
        i = ij.y;
        j = ij.x;
    }

    enum SHELL_TYPE {CLOSED, OPEN, VIRTUAL};
    SHELL_TYPE shell_i, shell_j;
    if(i < num_closed) shell_i = CLOSED;
    else if(i < num_closed+num_open) shell_i = OPEN;
    else shell_i = VIRTUAL;
    if(j < num_closed) shell_j = CLOSED;
    else if(j < num_closed+num_open) shell_j = OPEN;
    else shell_j = VIRTUAL;

    const auto Acc = rohf_params.Acc;
    const auto Bcc = rohf_params.Bcc;
    const auto Aoo = rohf_params.Aoo;
    const auto Boo = rohf_params.Boo;
    const auto Avv = rohf_params.Avv;
    const auto Bvv = rohf_params.Bvv;

    real_t d = 0.0;

    if(shell_i == CLOSED && shell_j == CLOSED){ // closed-closed
        d = 2.0 * (Acc*d_fock_mo_open_matrix[i*num_basis+j] + Bcc*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }else if(shell_i == CLOSED && shell_j == OPEN){ // closed-open
        d = 2.0 * (d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]);
    }else if(shell_i == CLOSED && shell_j == VIRTUAL){ // closed-virtual
        d = d_fock_mo_closed_matrix[i*num_basis+j];
    }else if(shell_i == OPEN && shell_j == OPEN){ // open-open
        d = 2.0 * (Aoo*d_fock_mo_open_matrix[i*num_basis+j] + Boo*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }else if(shell_i == OPEN && shell_j == VIRTUAL){ // open-virtual
        d = 2.0 * d_fock_mo_open_matrix[i*num_basis+j];
    }else if(shell_i == VIRTUAL && shell_j == VIRTUAL){ // virtual-virtual
        d = 2.0 * (Avv*d_fock_mo_open_matrix[i*num_basis+j] + Bvv*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }

    // 2-fold symmetry
    d_unified_fock_matrix[i*num_basis+j] = d;
    if(i != j) d_unified_fock_matrix[j*num_basis+i] = d;
}



/**
 * @brief CUDA kernel for computing the trace of a matrix
 * @param d_matrix Device pointer to the matrix
 * @param d_trace Device pointer to the trace
 * @param num_basis Number of basis functions
 * @details This function computes the trace of a matrix.
 */
__global__ void getMatrixTrace(const double* d_matrix, double* d_trace, const int num_basis)
{
    if (threadIdx.x >= num_basis) return;

    __shared__ double s_trace;
    if (threadIdx.x == 0) {
        s_trace = 0;
    }
    __syncthreads();

    atomicAdd(&s_trace, d_matrix[num_basis * threadIdx.x + threadIdx.x]);
    __syncthreads();
    if (threadIdx.x == 0) {
        d_trace[0] = s_trace;
    }
}

/**
 * @brief CUDA kernel for computing the initial Fock matrix in GWH method
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_overlap_matrix Device pointer to the overlap matrix
 * @param d_fock_matrix Device pointer to the initial Fock matrix
 * @param num_basis Number of basis functions
 * @param c_x Constant c_x
 */
__global__ void computeInitialFockMatrix_GWH_kernel(const double* d_core_hamiltonian_matrix, const double* d_overlap_matrix, double* d_fock_matrix, const int num_basis, const double c_x) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t p = id / num_basis;
    size_t q = id % num_basis;

    d_fock_matrix[id] = c_x * d_overlap_matrix[id] * (d_core_hamiltonian_matrix[p*num_basis+p] + d_core_hamiltonian_matrix[q*num_basis+q]) / 2.0;
}


__global__ void computeRIIntermediateMatrixB_kernel(const double* d_three_center_eri, const double* d_matrix_L, double* d_matrix_B, const int num_basis, const int num_auxiliary_basis){
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_auxiliary_basis * num_basis * num_basis) return;

    const size_t p = id / (num_basis*num_basis);
    const size_t id2 = (id % (num_basis*num_basis)) ;
    const size_t mu = id2 / num_basis;
    const size_t nu = id2 % num_basis;

    real_t sum = 0.0;
    for (int q = 0; q < num_auxiliary_basis; q++) {
        sum += d_three_center_eri[q*num_basis*num_basis + mu*num_basis + nu] * d_matrix_L[q*num_auxiliary_basis + p];
    }
    d_matrix_B[id] = sum;
}



__global__ void computeFockMatrix_RI_RHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - 0.5*d_K_matrix[id];
}


__global__ void computeFockMatrix_RI_UHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - d_K_matrix[id];
}

__global__ void computeFockMatrix_RI_ROHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix_closed, const double* d_K_matrix_open, double* d_Fock_matrix_closed, double* d_Fock_matrix_open, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix_closed[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - 0.5*d_K_matrix_closed[id];
    d_Fock_matrix_open[id] = 0.5 * (d_core_hamiltonian_matrix[id] + d_J_matrix[id] - d_K_matrix_open[id]);
}

/*
 * @brief Sets zeros to the upper triangular part of the matrix
 *
 * @param d_A Pointer to the N x N matrix in device memory (input/output).
 * @param N The size of the matrix (number of rows/columns).
 */
 __global__ void setZeroUpperTriangle(double* d_A, const int N) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = id / N;
    const size_t col = id % N;
    if (row < N && col < N && col > row) {
        d_A[row * N + col] = 0.0;
    }
}


/**
 * @brief CUDA kernel for computing the diagonal of the product of two matrices A and B
 * @param A Device pointer to the first matrix (row-major)
 * @param B Device pointer to the second matrix (row-major)
 * @param diag Device pointer to the output diagonal vector
 * @param N Size of the matrices (N x N)
 */
__global__ void compute_diagonal_of_product(const double* A, const double* B, double* diag, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + i];  // Diagonal element of the product matrix stored in row-major order
        }
        diag[i] = sum;
    }
}


/**
 * @brief CUDA kernel for computing the diagonal of the sum of two matrices A and B, multiplied by a third matrix C
 * @param A Device pointer to the first matrix (row-major)
 * @param B Device pointer to the second matrix (row-major)
 * @param C Device pointer to the third matrix (row-major)
 * @param diag Device pointer to the output diagonal vector
 * @param N Size of the matrices (N x N)
 */
__global__ void compute_diagonal_of_product_sum(const double* A, const double* B, const double* C, double* diag, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        double a_plus_b = A[i * N + k] + B[i * N + k]; // (A + B)[i][k]
        double c = C[k * N + i];                       // C[k][i] (row-major)
        sum += a_plus_b * c;
    }
    diag[i] = sum;
}



__global__ void constructERIHash_kernel(const std::vector<ShellTypeInfo> shell_type_infos, const std::vector<ShellPairTypeInfo> shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors, /* Hash memoryへのポインタ, */ const bool verbose)
{
    // ここにERIを計算して、ハッシュテーブルに格納する処理を実装する
}

__global__ void computeFockMatrix_Hash_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, /* Hash memoryへのポインタ, */ real_t* d_fock_matrix, const int num_basis, const int verbose)
{
    // ハッシュテーブルを使用してFock行列を計算する処理を実装する
}

__global__ void computeFockMatrix_DFT_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    int mu = id / num_basis, nu = id % num_basis;
    size_t idx_JK = (mu <= nu) ? mu*num_basis + nu : nu*num_basis+mu;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[idx_JK] + d_K_matrix[idx_JK];
}











inline __device__
void swap_indices(int& mu, int& nu)
{
    int tmp = mu;
    mu = nu;
    nu = tmp;
}

inline __device__
int utm_id(int mu, int nu) {
    return (mu <= nu) ? nu * (nu + 1) / 2 + mu : mu * (mu + 1) / 2 + nu;
}


inline __device__
int twoDim2oneDim(int mu, int nu, int num_basis)
{
    return (mu < nu) ? num_basis * mu + nu : num_basis * nu + mu;
}





inline __device__
void add2J(double val, double* g_J, 
              int mu, int nu, int la, int si, int num_basis, const double* g_dens) 
{
    if (mu > nu) {
        swap_indices(mu, nu);
    }
    if (la > si) {
        swap_indices(la, si);
    }
    if (mu > la || (mu == la && nu > si)) {
        swap_indices(mu, la);
        swap_indices(nu, si);
    }

    bool is_sym_bra = false;
    bool is_sym_ket = false;
    bool is_sym_braket = false;
    if (mu == nu) is_sym_bra = true;
    if (la == si) is_sym_ket = true;
    if (mu == la && nu == si) is_sym_braket = true;

    if (is_sym_bra && is_sym_ket && is_sym_braket) {
        atomicAdd(g_J + num_basis * mu + nu, 0.5 * g_dens[num_basis * la + si] * val);
    }
    else if (is_sym_bra && is_sym_ket) {
        atomicAdd(g_J + num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_J + num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        // atomicAdd(g_J + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
    }
    else if (is_sym_bra) {
        atomicAdd(g_J + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_J + num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        // atomicAdd(g_J + num_basis * mu + la, ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        // atomicAdd(g_J + num_basis * nu + si, (-0.5) * g_dens[num_basis * mu + la] * val);
    }
    else if (is_sym_ket) {
        atomicAdd(g_J + num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_J + num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        // atomicAdd(g_J + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
        // atomicAdd(g_J + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
    }
    else if (is_sym_braket) {
        atomicAdd(g_J + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        // atomicAdd(g_J + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
        // atomicAdd(g_J + num_basis * nu + si, (-0.5) * g_dens[num_basis * mu + la] * val);
        // atomicAdd(g_J + num_basis * mu + si, (-0.5) * g_dens[num_basis * nu + la] * val);
    }
    else {
        atomicAdd(g_J + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_J + num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        // atomicAdd(g_J + num_basis * mu + la, ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        // atomicAdd(g_J + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
        // atomicAdd(g_J + num_basis * mu + si, (-0.5) * g_dens[num_basis * nu + la] * val);
        // atomicAdd(g_J + ((nu <= la) ? num_basis * nu + la : num_basis * la + nu), ((nu == la) ? -1.0 : -0.5) * g_dens[num_basis * mu + si] * val);
    }
}





inline __device__
void add2K(double val, double* g_K, 
              int mu, int nu, int la, int si, int num_basis, const double* g_dens) 
{
    if (mu > nu) {
        swap_indices(mu, nu);
    }
    if (la > si) {
        swap_indices(la, si);
    }
    if (mu > la || (mu == la && nu > si)) {
        swap_indices(mu, la);
        swap_indices(nu, si);
    }

    bool is_sym_bra = false;
    bool is_sym_ket = false;
    bool is_sym_braket = false;
    if (mu == nu) is_sym_bra = true;
    if (la == si) is_sym_ket = true;
    if (mu == la && nu == si) is_sym_braket = true;

    if (is_sym_bra && is_sym_ket && is_sym_braket) {
        // atomicAdd(g_K + num_basis * mu + nu, 0.5 * g_dens[num_basis * la + si] * val);
    }
    else if (is_sym_bra && is_sym_ket) {
        // atomicAdd(g_K + num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        // atomicAdd(g_K + num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_K + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
    }
    else if (is_sym_bra) {
        // atomicAdd(g_K + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        // atomicAdd(g_K + num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_K + num_basis * mu + la, ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_K + num_basis * nu + si, (-0.5) * g_dens[num_basis * mu + la] * val);
    }
    else if (is_sym_ket) {
        // atomicAdd(g_K + num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        // atomicAdd(g_K + num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_K + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_K + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
    }
    else if (is_sym_braket) {
        // atomicAdd(g_K + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_K + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_K + num_basis * nu + si, (-0.5) * g_dens[num_basis * mu + la] * val);
        atomicAdd(g_K + num_basis * mu + si, (-0.5) * g_dens[num_basis * nu + la] * val);
    }
    else {
        // atomicAdd(g_K + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        // atomicAdd(g_K + num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_K + num_basis * mu + la, ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_K + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
        atomicAdd(g_K + num_basis * mu + si, (-0.5) * g_dens[num_basis * nu + la] * val);
        atomicAdd(g_K + ((nu <= la) ? num_basis * nu + la : num_basis * la + nu), ((nu == la) ? -1.0 : -0.5) * g_dens[num_basis * mu + si] * val);
    }
}






__global__ void MD_1T1SP_Direct_J(double* g_J, const double* g_density_matrix, const PrimitiveShell* g_shell, const size_t2* d_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, 
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads, const real_t swartz_screening_threshold, const double* g_upper_bound_factors,
    const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket)
{
    // 通し番号indexの計算
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;


    if (id >= num_threads) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;
    
    // Compute 4D index from thread id
    int ket_size;
    if(shell_s2.start_index == shell_s3.start_index){
        ket_size = (shell_s2.count * (shell_s2.count+1)) / 2;
    }else{
        ket_size = shell_s2.count*shell_s3.count;
    }
    const size_t2 abcd = index1to2(id, (shell_s0.start_index == shell_s2.start_index && shell_s1.start_index == shell_s3.start_index), ket_size);
    // const size_t2 ab = index1to2(abcd.x, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    // const size_t2 cd = index1to2(abcd.y, shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    // Task-wise Schwarz screening
    if (g_upper_bound_factors[head_bra + abcd.x] * g_upper_bound_factors[head_ket + abcd.y] < swartz_screening_threshold) {
        return;
    }

    // Obtain primitive shells [ab|cd]
    const size_t primitive_index_a = d_primitive_shell_pair_indices[head_bra + abcd.x].x + shell_s0.start_index;
	const size_t primitive_index_b = d_primitive_shell_pair_indices[head_bra + abcd.x].y + shell_s1.start_index;
    const size_t primitive_index_c = d_primitive_shell_pair_indices[head_ket + abcd.y].x + shell_s2.start_index;
	const size_t primitive_index_d = d_primitive_shell_pair_indices[head_ket + abcd.y].y + shell_s3.start_index;
    // const size_t primitive_index_a = ab.x+shell_s0.start_index;
    // const size_t primitive_index_b = ab.y+shell_s1.start_index;
    // const size_t primitive_index_c = cd.x+shell_s2.start_index;
    // const size_t primitive_index_d = cd.y+shell_s3.start_index;

    const PrimitiveShell a = g_shell[primitive_index_a];
    const PrimitiveShell b = g_shell[primitive_index_b];
    const PrimitiveShell c = g_shell[primitive_index_c];
    const PrimitiveShell d = g_shell[primitive_index_d];
        
    // Obtain basis index (ij|kl)
    const size_t size_a = a.basis_index;
    const size_t size_b = b.basis_index;
    const size_t size_c = c.basis_index;
    const size_t size_d = d.basis_index;

    bool is_bra_symmetric = (primitive_index_a == primitive_index_b);
    bool is_ket_symmetric = (primitive_index_c == primitive_index_d);
    bool is_braket_symmetric = (primitive_index_a == primitive_index_c && primitive_index_b == primitive_index_d);
    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double delta = d.exponent;
    const double p = alpha+beta;
    const double q = gamma+delta;
    const double xi = p*q / (p+q);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;
    const double coef_d = d.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_D[3] = {d.coordinate.x, d.coordinate.y, d.coordinate.z};

    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};
    const double pos_Q[3] = {(gamma*pos_C[0]+delta*pos_D[0])/(gamma+delta), (gamma*pos_C[1]+delta*pos_D[1])/(gamma+delta), (gamma*pos_C[2]+delta*pos_D[2])/(gamma+delta)};

    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;
    const int orbital_D = d.shell_type;

    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_Q[0])*(pos_P[0]-pos_Q[0]) + (pos_P[1]-pos_Q[1])*(pos_P[1]-pos_Q[1]) + (pos_P[2]-pos_Q[2])*(pos_P[2]-pos_Q[2]));


    const int K=orbital_A + orbital_B + orbital_C + orbital_D;
    
    double Boys[25];
    getIncrementalBoys(K, xi*dist, g_boys_grid, Boys);

    //Boys関数の値を計算(Single)
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2*xi), i));
    }

    //各ERIを計算
    //事前計算⇒実際のERI計算の順に実行
    //p軌道の場合lmn_aが0:px, 1:py, 2:pz軌道のように対応付け
    //d以上はconstant配列のloop_to_angを参照

    
    
    double Norm_A, Norm_B, Norm_C, Norm_D;
    double Norm;

    int t,u,v,tau,nu,phi;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    int tid=0;
    
    int iter_max;


    // 方位量子数l,m,nの値をループ変数から導出
    for(int lmn_a=0; lmn_a<comb_max(orbital_A); lmn_a++){
        int l1=loop_to_ang[orbital_A][lmn_a][0]; int m1=loop_to_ang[orbital_A][lmn_a][1]; int n1=loop_to_ang[orbital_A][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);


        for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){                  
            int l2=loop_to_ang[orbital_B][lmn_b][0]; int m2=loop_to_ang[orbital_B][lmn_b][1]; int n2=loop_to_ang[orbital_B][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);

            for(int lmn_c=0; lmn_c<comb_max(orbital_C); lmn_c++){
                int l3=loop_to_ang[orbital_C][lmn_c][0]; int m3=loop_to_ang[orbital_C][lmn_c][1]; int n3=loop_to_ang[orbital_C][lmn_c][2];
                Norm_C = calcNorm(gamma, l3, m3, n3);


                for(int lmn_d=0; lmn_d<comb_max(orbital_D); lmn_d++){
                    int l4=loop_to_ang[orbital_D][lmn_d][0]; int m4=loop_to_ang[orbital_D][lmn_d][1]; int n4=loop_to_ang[orbital_D][lmn_d][2];
                    Norm_D = calcNorm(delta, l4, m4, n4);
                
                    if(orbital_A==orbital_C && orbital_C==0 && orbital_B==orbital_D && orbital_D==1 && (size_a==size_c && size_b==size_d) && lmn_b > lmn_d) continue; //spsp
                    if(orbital_A==0 && orbital_C==orbital_D && orbital_D==1 && size_c==size_d && lmn_c > lmn_d) continue; // sspp, sppp
                    if(orbital_A==orbital_B && orbital_B==orbital_C && orbital_C==orbital_D && orbital_D==1){ //
                        if(size_c==size_d && lmn_c > lmn_d) continue;
                        if(size_a==size_b && lmn_a > lmn_b) continue;
                    }

                    Norm = Norm_A * Norm_B * Norm_C * Norm_D;
                    // 前回のループの計算結果をクリア
                    thread_val=0.0;
                    // 事前計算部
                    //初期値：Boysとして計算済
                    //Step 0: Boys関数評価
                    R[0]=Boys[0];
                    for(int i=0; i <= K; i++){
                        R_mid[i]=Boys[i];
                    }
                    
                    // ループ変数の設定
                    t_max = l1+l2+1;
                    u_max = m1+m2+1;
                    v_max = n1+n2+1;
                    tau_max = l3+l4+1;
                    nu_max = m3+m4+1;
                    phi_max = n3+n4+1;

                    for(int k=1; k <= K; k++){//Step 1~Kの計算
                        // t+u+v=kとなる全ペアに対して適切な計算
                        // 0~K-kまでそれぞれ必要⇒ループでやる
        
        
                        for(int z=0; z<=(K+1)*comb_max(k); z++){
                        
                            int i = z/comb_max(k);
        
                            if(i <= K-k){
                                t=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                                u=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                                v=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
        
                                if((t <= (t_max+tau_max-2)) && (u <= (u_max+nu_max-2)) && (v <= (v_max+phi_max-2))){
                                    if(t >= 1){
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_Q[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                    else if(u >= 1){
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_Q[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                    else{
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_Q[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                }
                            }
                        }//step kの全計算が終了
        

                        //必要な結果を配列Rに書き込み
                        for(int i=0; i<=comb_max(k); i++){
                            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_Rmid) + i];
                        }

                    }
                    //事前計算完了
                    

                    // ERI計算部
                    iter_max=t_max*u_max*v_max*tau_max*nu_max*phi_max + 1;
                    for(int i=0; i<iter_max; i++){
                        // MD法6重ループを管理する6変数を各Threadに割り当て
                        tid=i;
                        phi = tid % phi_max;
                        tid /= phi_max;
                        nu = tid % nu_max;
                        tid /= nu_max;
                        tau = tid % tau_max;
                        tid /= tau_max;
                        v = tid % v_max;
                        tid /= v_max;
                        u = tid % u_max;
                        tid /= u_max;
                        t=tid;


                        double my_val = 0.0;

                        ////特定の(t,u,v,tau,nu,phi)に対応する結果をmy_valとして持つ
                        if(t <= t_max-1 && u<=u_max-1 && v<=v_max-1 && tau<=tau_max-1 && nu<=nu_max-1 && phi<=phi_max-1){
                            int k=t+u+v+tau+nu+phi;
                            my_val = MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0])) * MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1])) * MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2])) * MD_Et_NonRecursion(l3, l4, tau, gamma, delta, (pos_C[0]-pos_D[0])) * MD_Et_NonRecursion(m3, m4, nu, gamma, delta, (pos_C[1]-pos_D[1])) * MD_Et_NonRecursion(n3, n4, phi, gamma, delta, (pos_C[2]-pos_D[2])) * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                            // thread_valに足しこんでMD法の結果を得る
                            thread_val += my_val*2 * M_PI_2_5 /(p*q * sqrt((p+q)))  *coef_a*coef_b*coef_c*coef_d;
                        } 
                    }

                    thread_val *= Norm * g_cgto_normalization_factors[size_a + lmn_a]
                                        * g_cgto_normalization_factors[size_b + lmn_b]
                                        * g_cgto_normalization_factors[size_c + lmn_c]
                                        * g_cgto_normalization_factors[size_d + lmn_d];



                    if(!is_bra_symmetric && size_a == size_b) thread_val *= 2.0;
                    if(!is_ket_symmetric && size_c == size_d) thread_val *= 2.0;
                    if(!is_braket_symmetric && (size_a==size_c && size_b==size_d)) thread_val *= 2.0;
                    if(orbital_A==orbital_B && orbital_B==orbital_C && orbital_C==orbital_D && orbital_D==1 && (utm_id(size_a,size_b) == utm_id(size_c,size_d)) && twoDim2oneDim(size_a+lmn_a,size_b+lmn_b,num_basis) != twoDim2oneDim(size_c+lmn_c,size_d+lmn_d,num_basis) ) thread_val *= 0.5;
                    
                    add2J(thread_val, g_J, size_a + lmn_a, size_b + lmn_b, size_c + lmn_c, size_d + lmn_d, num_basis, g_density_matrix);

                }
            }
        }
    }
    return;
}














__global__ void MD_1T1SP_Direct_K(double* g_K, const double* g_density_matrix, const PrimitiveShell* g_shell, const size_t2* d_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, 
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads, const real_t swartz_screening_threshold, const double* g_upper_bound_factors,
    const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket)
{
    // 通し番号indexの計算
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;


    if (id >= num_threads) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;
    
    // Compute 4D index from thread id
    int ket_size;
    if(shell_s2.start_index == shell_s3.start_index){
        ket_size = (shell_s2.count * (shell_s2.count+1)) / 2;
    }else{
        ket_size = shell_s2.count*shell_s3.count;
    }
    // const size_t2 abcd = index1to2(id, false, ket_size);
    const size_t2 abcd = index1to2(id, (shell_s0.start_index == shell_s2.start_index && shell_s1.start_index == shell_s3.start_index), ket_size);
    // const size_t2 ab = index1to2(abcd.x, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    // const size_t2 cd = index1to2(abcd.y, shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    // Task-wise Schwarz screening
    if (g_upper_bound_factors[head_bra + abcd.x] * g_upper_bound_factors[head_ket + abcd.y] < swartz_screening_threshold) {
        return;
    }

    // Obtain primitive shells [ab|cd]
    const size_t primitive_index_a = d_primitive_shell_pair_indices[head_bra + abcd.x].x + shell_s0.start_index;
	const size_t primitive_index_b = d_primitive_shell_pair_indices[head_bra + abcd.x].y + shell_s1.start_index;
    const size_t primitive_index_c = d_primitive_shell_pair_indices[head_ket + abcd.y].x + shell_s2.start_index;
	const size_t primitive_index_d = d_primitive_shell_pair_indices[head_ket + abcd.y].y + shell_s3.start_index;
    // const size_t primitive_index_a = ab.x+shell_s0.start_index;
    // const size_t primitive_index_b = ab.y+shell_s1.start_index;
    // const size_t primitive_index_c = cd.x+shell_s2.start_index;
    // const size_t primitive_index_d = cd.y+shell_s3.start_index;

    const PrimitiveShell a = g_shell[primitive_index_a];
    const PrimitiveShell b = g_shell[primitive_index_b];
    const PrimitiveShell c = g_shell[primitive_index_c];
    const PrimitiveShell d = g_shell[primitive_index_d];
        
    // Obtain basis index (ij|kl)
    const size_t size_a = a.basis_index;
    const size_t size_b = b.basis_index;
    const size_t size_c = c.basis_index;
    const size_t size_d = d.basis_index;

    bool is_bra_symmetric = (primitive_index_a == primitive_index_b);
    bool is_ket_symmetric = (primitive_index_c == primitive_index_d);
    bool is_braket_symmetric = (primitive_index_a == primitive_index_c && primitive_index_b == primitive_index_d);
    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double delta = d.exponent;
    const double p = alpha+beta;
    const double q = gamma+delta;
    const double xi = p*q / (p+q);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;
    const double coef_d = d.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_D[3] = {d.coordinate.x, d.coordinate.y, d.coordinate.z};

    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};
    const double pos_Q[3] = {(gamma*pos_C[0]+delta*pos_D[0])/(gamma+delta), (gamma*pos_C[1]+delta*pos_D[1])/(gamma+delta), (gamma*pos_C[2]+delta*pos_D[2])/(gamma+delta)};

    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;
    const int orbital_D = d.shell_type;

    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_Q[0])*(pos_P[0]-pos_Q[0]) + (pos_P[1]-pos_Q[1])*(pos_P[1]-pos_Q[1]) + (pos_P[2]-pos_Q[2])*(pos_P[2]-pos_Q[2]));


    const int K=orbital_A + orbital_B + orbital_C + orbital_D;
    
    double Boys[25];
    getIncrementalBoys(K, xi*dist, g_boys_grid, Boys);

    //Boys関数の値を計算(Single)
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2*xi), i));
    }

    //各ERIを計算
    //事前計算⇒実際のERI計算の順に実行
    //p軌道の場合lmn_aが0:px, 1:py, 2:pz軌道のように対応付け
    //d以上はconstant配列のloop_to_angを参照

    
    
    double Norm_A, Norm_B, Norm_C, Norm_D;
    double Norm;

    int t,u,v,tau,nu,phi;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    int tid=0;
    
    int iter_max;


    // 方位量子数l,m,nの値をループ変数から導出
    for(int lmn_a=0; lmn_a<comb_max(orbital_A); lmn_a++){
        int l1=loop_to_ang[orbital_A][lmn_a][0]; int m1=loop_to_ang[orbital_A][lmn_a][1]; int n1=loop_to_ang[orbital_A][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);


        for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){                  
            int l2=loop_to_ang[orbital_B][lmn_b][0]; int m2=loop_to_ang[orbital_B][lmn_b][1]; int n2=loop_to_ang[orbital_B][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);

            for(int lmn_c=0; lmn_c<comb_max(orbital_C); lmn_c++){
                int l3=loop_to_ang[orbital_C][lmn_c][0]; int m3=loop_to_ang[orbital_C][lmn_c][1]; int n3=loop_to_ang[orbital_C][lmn_c][2];
                Norm_C = calcNorm(gamma, l3, m3, n3);


                for(int lmn_d=0; lmn_d<comb_max(orbital_D); lmn_d++){
                    int l4=loop_to_ang[orbital_D][lmn_d][0]; int m4=loop_to_ang[orbital_D][lmn_d][1]; int n4=loop_to_ang[orbital_D][lmn_d][2];
                    Norm_D = calcNorm(delta, l4, m4, n4);
                
                    if(orbital_A==orbital_C && orbital_C==0 && orbital_B==orbital_D && orbital_D==1 && (size_a==size_c && size_b==size_d) && lmn_b > lmn_d) continue; //spsp
                    if(orbital_A==0 && orbital_C==orbital_D && orbital_D==1 && size_c==size_d && lmn_c > lmn_d) continue; // sspp, sppp
                    if(orbital_A==orbital_B && orbital_B==orbital_C && orbital_C==orbital_D && orbital_D==1){ //
                        if(size_c==size_d && lmn_c > lmn_d) continue;
                        if(size_a==size_b && lmn_a > lmn_b) continue;
                    }

                    Norm = Norm_A * Norm_B * Norm_C * Norm_D;
                    // 前回のループの計算結果をクリア
                    thread_val=0.0;
                    // 事前計算部
                    //初期値：Boysとして計算済
                    //Step 0: Boys関数評価
                    R[0]=Boys[0];
                    for(int i=0; i <= K; i++){
                        R_mid[i]=Boys[i];
                    }
                    
                    // ループ変数の設定
                    t_max = l1+l2+1;
                    u_max = m1+m2+1;
                    v_max = n1+n2+1;
                    tau_max = l3+l4+1;
                    nu_max = m3+m4+1;
                    phi_max = n3+n4+1;

                    for(int k=1; k <= K; k++){//Step 1~Kの計算
                        // t+u+v=kとなる全ペアに対して適切な計算
                        // 0~K-kまでそれぞれ必要⇒ループでやる
        
        
                        for(int z=0; z<=(K+1)*comb_max(k); z++){
                        
                            int i = z/comb_max(k);
        
                            if(i <= K-k){
                                t=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                                u=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                                v=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
        
                                if((t <= (t_max+tau_max-2)) && (u <= (u_max+nu_max-2)) && (v <= (v_max+phi_max-2))){
                                    if(t >= 1){
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_Q[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                    else if(u >= 1){
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_Q[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                    else{
                                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_Q[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
                                    }
                                }
                            }
                        }//step kの全計算が終了
        

                        //必要な結果を配列Rに書き込み
                        for(int i=0; i<=comb_max(k); i++){
                            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_Rmid) + i];
                        }

                    }
                    //事前計算完了
                    

                    // ERI計算部
                    iter_max=t_max*u_max*v_max*tau_max*nu_max*phi_max + 1;
                    for(int i=0; i<iter_max; i++){
                        // MD法6重ループを管理する6変数を各Threadに割り当て
                        tid=i;
                        phi = tid % phi_max;
                        tid /= phi_max;
                        nu = tid % nu_max;
                        tid /= nu_max;
                        tau = tid % tau_max;
                        tid /= tau_max;
                        v = tid % v_max;
                        tid /= v_max;
                        u = tid % u_max;
                        tid /= u_max;
                        t=tid;


                        double my_val = 0.0;

                        ////特定の(t,u,v,tau,nu,phi)に対応する結果をmy_valとして持つ
                        if(t <= t_max-1 && u<=u_max-1 && v<=v_max-1 && tau<=tau_max-1 && nu<=nu_max-1 && phi<=phi_max-1){
                            int k=t+u+v+tau+nu+phi;
                            my_val = MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0])) * MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1])) * MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2])) * MD_Et_NonRecursion(l3, l4, tau, gamma, delta, (pos_C[0]-pos_D[0])) * MD_Et_NonRecursion(m3, m4, nu, gamma, delta, (pos_C[1]-pos_D[1])) * MD_Et_NonRecursion(n3, n4, phi, gamma, delta, (pos_C[2]-pos_D[2])) * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                            // thread_valに足しこんでMD法の結果を得る
                            thread_val += my_val*2 * M_PI_2_5 /(p*q * sqrt((p+q)))  *coef_a*coef_b*coef_c*coef_d;
                        } 
                    }

                    thread_val *= Norm * g_cgto_normalization_factors[size_a + lmn_a]
                                        * g_cgto_normalization_factors[size_b + lmn_b]
                                        * g_cgto_normalization_factors[size_c + lmn_c]
                                        * g_cgto_normalization_factors[size_d + lmn_d];



                    if(!is_bra_symmetric && size_a == size_b) thread_val *= 2.0;
                    if(!is_ket_symmetric && size_c == size_d) thread_val *= 2.0;
                    if(!is_braket_symmetric && (size_a==size_c && size_b==size_d)) thread_val *= 2.0;
                    if(orbital_A==orbital_B && orbital_B==orbital_C && orbital_C==orbital_D && orbital_D==1 && (utm_id(size_a,size_b) == utm_id(size_c,size_d)) && twoDim2oneDim(size_a+lmn_a,size_b+lmn_b,num_basis) != twoDim2oneDim(size_c+lmn_c,size_d+lmn_d,num_basis) ) thread_val *= 0.5;
                    
                    add2K(thread_val, g_K, size_a + lmn_a, size_b + lmn_b, size_c + lmn_c, size_d + lmn_d, num_basis, g_density_matrix);

                }
            }
        }
    }
    return;
}


__global__ void get_rho_kernel(int nao,
                               int ngrids,
                               const double *dm,   // (nao,nao)
                               const double *ao,   // (ngrids,nao)
                               double *rho_out)    // (ngrids)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrids) return;

    const double *phi_g = ao + (size_t)g * nao;
    double r = 0.0;
    for (int u = 0; u < nao; ++u) {
        double phiu = phi_g[u];
        const double *dm_row = dm + (size_t)u * nao;
        for (int v = 0; v < nao; ++v)
            r += dm_row[v] * phiu * phi_g[v];
    }
    rho_out[g] = r;
}

struct VWNPar {
    double A, b, c, x0;
};
static const VWNPar vwn_param_host[2] = {
    {0.0310907,  3.72744, 12.9352, -0.10498},   // ζ=0
    {0.01554535, 7.06042, 18.0578, -0.32500}    // ζ=1
};

/* For device use, copy parameters to constant memory */
__constant__ VWNPar vwn_param[2];

/* ---------- device double atomicAdd fallback ---------- */
__device__ inline double atomicAdd_double(double *address, double val) {
#if __CUDA_ARCH__ >= 600
    // On modern architectures, use hardware atomicAdd for double
    return atomicAdd(address, val);
#else
    // Fallback implementation using atomicCAS on 64-bit integer representation
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    double old_val;
    do {
        assumed = old;
        old_val = __longlong_as_double(assumed);
        unsigned long long int new_val_ull = __double_as_longlong(old_val + val);
        old = atomicCAS(address_as_ull, assumed, new_val_ull);
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

/* ---------- 1b. device math helpers ---------- */
__device__ inline void vwn_ec_device(double x, const VWNPar &p, double &ec, double &dec_dx)
{
    const double X = x * x + p.b * x + p.c;
    const double Q = sqrt(4.0 * p.c - p.b * p.b);
    const double log_term  = log(x * x / X);
    const double atan_term = 2.0 * p.b / Q * atan(Q / (2.0 * x + p.b));
    const double x02 = p.x0 * p.x0;
    const double denom = x02 + p.b * p.x0 + p.c;
    const double corr  = p.b * p.x0 / denom *
        (log((x - p.x0) * (x - p.x0) / X) +
         2.0 * (2.0 * p.x0 + p.b) / Q * atan(Q / (2.0 * x + p.b)));
    ec = p.A * (log_term + atan_term - corr);
    dec_dx = p.A * (2.0 / x - (2.0 * x + p.b) / X -
                    p.b * p.x0 / denom * (2.0 / (x - p.x0) - (2.0 * x + p.b) / X));
}


__global__ void lda_exc_vxc_kernel(int ngrid,
                                   const double *rho,
                                   double *exc,
                                   double *vxc,
                                   double  zeta)
{
    const double pi = 3.14159265358979323846;
    const double Cx = 0.7385587663820224;

    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrid) return;

    double r = rho[g];
    if (r < 1e-300) r = 1e-300;
    double rs = pow(3.0 / (4.0 * pi * r), 1.0 / 3.0);
    double x  = sqrt(rs);

    double ec0, dec0_dx, ec1, dec1_dx;
    vwn_ec_device(x, vwn_param[0], ec0, dec0_dx);
    vwn_ec_device(x, vwn_param[1], ec1, dec1_dx);

    double z2 = zeta * zeta;
    double ec     = ec0 + (ec1 - ec0) * z2;
    double dec_dx = dec0_dx + (dec1_dx - dec0_dx) * z2;
    double vc     = ec - rs / 3.0 * dec_dx / (2.0 * x);

    double rho13 = pow(r, 1.0 / 3.0);
    double ex    = -Cx * r * rho13;
    double vx    = -4.0 / 3.0 * Cx * rho13;

    if (exc) exc[g] = ex + r * ec;
    if (vxc) vxc[g] = vx + vc;
}

__global__ void build_vxc_matrix_kernel(int nao,
                                        int rows,        
                                        int g0,         
                                        const double *ao_b,   
                                        const double *w_b,    
                                        const double *vxc_b,  
                                        double *vxc_mat)      
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nao) return;

    int im = idx / nao;         
    int i  = idx % nao;          
    int g  = g0 + im;            

    double aoi = ao_b[im * nao + i];
    double w   = w_b[im];
    double vxc = vxc_b[im];

    for (int j = 0; j < nao; ++j) {
        double aoj = ao_b[im * nao + j];
        double contrib = w * vxc * aoi * aoj;
        atomicAdd_double(&vxc_mat[i * nao + j], contrib);
    }
}
} // namespace gpu