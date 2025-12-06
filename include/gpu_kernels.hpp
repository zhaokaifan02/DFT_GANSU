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



#pragma once

#include "types.hpp"
#include "utils_cuda.hpp"

#include "int2e.hpp"
#include "utils.hpp" // THROW_EXCEPTION
#include "int2c2e.hpp"
#include "int3c2e.hpp"
#include "Et_functions.hpp"

namespace gansu::gpu {

// constant values
const int WARP_SIZE = 32;
const unsigned int FULL_MASK = 0xffffffff;


// prototype declarations of CUDA kernels
__global__ void inverseSqrt_kernel(double* d_eigenvalues, const size_t size, const double threshold);
__global__ void computeDensityMatrix_RHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix, const int num_electron, const size_t num_basis);
__global__ void computeDensityMatrix_UHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix, const int num_spin, const size_t num_basis);
__global__ void computeDensityMatrix_ROHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix_closed, double* d_density_matrix_open, double* d_density_matrix, const int num_closed, const int num_open, const size_t num_basis);
__global__ void transposeMatrixInPlace_kernel(double* d_matrix, int size);
__global__ void computeFockMatrix_RHF_kernel(const double* d_density_matrix, const double* d_core_hamiltonian_matrix, const double* d_eri, double* d_fock_matrix, int num_basis);
__global__ void computeFockMatrix_UHF_kernel(const double* d_density_matrix_a, const double* d_density_matrix_b, const double* d_core_hamiltonian_matrix, const double* d_eri, double* d_fock_matrix_a, double* d_fock_matrix_b, int num_basis);
__global__ void computeFockMatrix_ROHF_kernel(const double* d_density_matrix_closed, const double* d_density_matrix_open, const double* d_core_hamiltonian_matrix, const double* d_eri, double* d_fock_matrix_closed, double* d_fock_matrix_open, int num_basis);
__global__ void computeUnifiedFockMatrix_ROHF_kernel(const double* d_fock_mo_closed_matrix, const double* d_fock_mo_open_matrix, const ROHF_ParameterSet rohf_params, double* d_unified_fock_matrix, const int num_closed, const int num_open, const size_t num_basis);
__global__ void getMatrixTrace(const double* d_matrix, double* d_trace, const int num_basis);
__global__ void computeInitialFockMatrix_GWH_kernel(const double* d_core_hamiltonian_matrix, const double* d_overlap_matrix, double* d_fock_matrix, const int num_basis, const double c_x);
__global__ void computeRIIntermediateMatrixB_kernel(const double* d_three_center_eri, const double* d_matrix_L, double* d_matrix_B, const int num_basis, const int num_auxiliary_basis);
__global__ void weighted_sum_matrices_kernel(double* d_J, const double* d_B, const double* d_W, const int M, const int N, const bool accumulated=false);
__global__ void sum_matrices_kernel(double* d_K, const double* d_B, const int M, const int N, const bool accumulated=false);
__global__ void computeFockMatrix_RI_RHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis);
__global__ void computeFockMatrix_RI_UHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis);
__global__ void computeFockMatrix_RI_ROHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix_closed, const double* d_K_matrix_open, double* d_Fock_matrix_closed, double* d_Fock_matrix_open, const int num_basis);
__global__ void setZeroUpperTriangle(double* d_A, const int N);
__global__ void compute_diagonal_of_product(const double* A, const double* B, double* diag, const int N);
__global__ void compute_diagonal_of_product_sum(const double* A, const double* B, const double* C, double* diag, const int N);

__global__ void constructERIHash_kernel(const std::vector<ShellTypeInfo> shell_type_infos, const std::vector<ShellPairTypeInfo> shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors, /* Hash memoryへのポインタ, */ const bool verbose);
__global__ void computeFockMatrix_Hash_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, /* Hash memoryへのポインタ, */ real_t* d_fock_matrix, const int num_basis, const int verbose);

__global__ void computeFockMatrix_DFT_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis);



__global__ void MD_1T1SP_Direct_K(double* g_K, const double* g_density_matrix, const PrimitiveShell* g_shell, const size_t2* d_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors,  const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t swartz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void MD_1T1SP_Direct_J(double* g_J, const double* g_density_matrix, const PrimitiveShell* g_shell, const size_t2* d_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors,  const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t swartz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);

__global__ void get_rho_kernel(int nao, int ngrids, const double *dm, const double *ao, double *rho_out);
__global__ void build_vxc_matrix_kernel(int nao, int rows, int g0, const double *ao_b, const double *w_b, const double *vxc_b, double *vxc_mat);
__global__ void lda_exc_vxc_kernel(int ngrid, const double *rho, double *exc, double *vxc, double  zeta);


} // namespace gansu::gpu