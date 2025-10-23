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


#ifndef INT2E_CUH
#define INT2E_CUH

#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"

#ifndef M_PI_2_5
    #define M_PI_2_5 17.49341832762486284622				
#endif

#ifndef PI3_4
    #define PI3_4 0.71270547035499016035339845
#endif

namespace gansu::gpu{

__global__ void ssss2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void sssp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void sspp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void spsp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void sppp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void pppp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void MD_1T1SP(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);

__global__ void get_schwarz_upper_bound_factors_ss(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_sp(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_pp(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_general(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_aux_general(const PrimitiveShell* g_shell_aux, const real_t* g_aux_cgto_normalization_factors, const ShellTypeInfo shell_s0, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors_aux);


// define the kernel functions as function pointers for two electron integrals
using eri_kernel_t = void (*)(double*, const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const size_t, const real_t, const double*, const int, const double*, const size_t, const size_t);
using schwarz_kernel_t = void (*)(const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const size_t, const size_t, const double*, double*);
using schwarz_aux_kernel_t = void (*)(const PrimitiveShell*, const real_t*, const ShellTypeInfo, const size_t, const size_t, const double*, double*);

// ここを対角，非対角の分岐にするぐらい？
inline eri_kernel_t get_eri_kernel(int a, int b, int c, int d){
    if(a>b) std::swap(a, b);
    if(c>d) std::swap(c, d);
    if(a>c || (a==c && b>d)){
        std::swap(a, c);
        std::swap(b, d);
    }
    
    //if(a==c && b==d) return UTM_1T1SP;
    //else return RCT_1T1SP;

    // if(a==0 && b==0 && c==0 && d==0)      return ssss2e;
    // else if(a==0 && b==0 && c==0 && d==1) return sssp2e;
    // else if(a==0 && b==0 && c==1 && d==1) return sspp2e;
    // else if(a==0 && b==1 && c==0 && d==1) return spsp2e;
    // else if(a==0 && b==1 && c==1 && d==1) return sppp2e;
    // else if(a==1 && b==1 && c==1 && d==1) return pppp2e;
    // else throw std::runtime_error("Invalid shell type");

    // Hybrid
    if (a == 0 && b == 0 && c == 0 && d == 0)      return ssss2e;
    else if (a == 0 && b == 0 && c == 0 && d == 1) return sssp2e;
    else if (a == 0 && b == 0 && c == 1 && d == 1) return sspp2e;
    else if (a == 0 && b == 1 && c == 0 && d == 1) return spsp2e;
    else if (a == 0 && b == 1 && c == 1 && d == 1) return sppp2e;
    else if (a == 1 && b == 1 && c == 1 && d == 1) return pppp2e;
    else return MD_1T1SP;
}

inline schwarz_kernel_t get_schwarz_kernel(int a, int b)
{
    if(a > b) std::swap(a, b);
    if (a == 0 && b == 0)       return get_schwarz_upper_bound_factors_ss;
    else if (a == 0 && b == 1)  return get_schwarz_upper_bound_factors_sp;
    else if (a == 1 && b == 1)  return get_schwarz_upper_bound_factors_pp;
    else                        return get_schwarz_upper_bound_factors_general;
}


inline schwarz_aux_kernel_t get_schwarz_aux_kernel(int a)
{
    return get_schwarz_upper_bound_factors_aux_general;
}



inline __device__
int factorial2_gpu(int n)
{
    return (n <= 1) ?  1 : n*factorial2_gpu(n-2);
}

inline __device__
long long int calcIdx4Dim(int i, int j, int k, int l, int nao)
{
    return nao * nao * nao * i + nao * nao * j + nao * k + l;
}

inline __device__
double calcNorm(double exp, int l, int m, int n){
    return static_cast<double>(1 << (l+m+n)) / static_cast<double>(sqrt(static_cast<double>(factorial2_gpu(2.0*l-1.0) * factorial2_gpu(2.0*m-1.0) * factorial2_gpu(2.0*n-1.0)))) * PI3_4 * pow(exp, (2.0*(l+m+n)+3.0)/4.0);
}

inline __device__
double calcNorms(double alpha, double beta, double gamma, double delta, int i1, int k1, int m1, int j1, int l1, int n1, int i2, int k2, int m2, int j2, int l2, int n2)
{
    return pow(2.0, i1+i2+j1+j2+k1+k2+l1+l2+m1+m2+n1+n2) 
           * pow(factorial2_gpu(2.0*i1-1.0)*factorial2_gpu(2.0*j1-1.0)*factorial2_gpu(2.0*k1-1.0)*factorial2_gpu(2.0*l1-1.0)*factorial2_gpu(2.0*m1-1.0)*factorial2_gpu(2.0*n1-1.0)*factorial2_gpu(2.0*i2-1.0)*factorial2_gpu(2.0*j2-1.0)*factorial2_gpu(2.0*k2-1.0)*factorial2_gpu(2.0*l2-1.0)*factorial2_gpu(2.0*m2-1.0)*factorial2_gpu(2.0*n2-1.0), -0.5) 
           * 8.0 / (M_PI*M_PI*M_PI)
           * pow(alpha, (2.0*(i1+k1+m1)+3.0)/4.0)
           * pow(beta, (2.0*(j1+l1+n1)+3.0)/4.0)
           * pow(gamma, (2.0*(i2+k2+m2)+3.0)/4.0)
           * pow(delta, (2.0*(j2+l2+n2)+3.0)/4.0);
}




/* case1. 全てひっくり返すか判定([ss|ss], [pp|pp])　*/
inline __device__
void addToResult_case1(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, bool sym_braket)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case2. bra側のみ判定 ([ss|sp]) */
inline __device__
void addToResult_case2(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case3. ket側のみ判定 ([sp|pp]) */
inline __device__
void addToResult_case3(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_ket)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case4. bra側ket側，それぞれ判定 ([ss|pp]) */
inline __device__
void addToResult_case4(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket)
{    
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    }
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    }
    if(!sym_bra && !sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case5. bra-ketを判定 ([sp|sp]) */
inline __device__
void addToResult_case5(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_braket)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}

/* case6. 判定しない ([sp|sd]) */
inline __device__
void addToResult_case6(double res, double *g_G, int p, int q, int r, int s, int nao)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
}


/* case1. 全てひっくり返すか判定([ss|ss], [pp|pp])　*/
inline __device__
void addToResult_case1(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, bool sym_braket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case2. bra側のみ判定 ([ss|sp]) */
inline __device__
void addToResult_case2(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case3. ket側のみ判定 ([sp|pp]) */
inline __device__
void addToResult_case3(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_ket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case4. bra側ket側，それぞれ判定 ([ss|pp]) */
inline __device__
void addToResult_case4(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, const double* g_cgto_normalization_factors)
{    
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    }
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    }
    if(!sym_bra && !sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case5. bra-ketを判定 ([sp|sp]) */
inline __device__
void addToResult_case5(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_braket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}

/* case6. 判定しない ([sp|sd]) */
inline __device__
void addToResult_case6(double res, double *g_G, int p, int q, int r, int s, int nao, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
}



inline __device__
void addToResult(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, bool sym_braket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];

    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}






// BatchアルゴリズムにおけるR_midへのindexを計算
inline __device__
int calc_Idx_Rmid(int k, int u, int v, int i, int comb_max, int size_mid){
    return ((k+3)%3)*size_mid + (i)*comb_max + v*(k+2) - v*(v+1)/2 + (u);
}

inline __device__
int comb_max(int k){
    return ((k+2)*(k+1))/2;
}

// ループ変数(2)⇒角運動量(0,0,1)への置換 総和6=i軌道まで対応可能
//順に1, 3, 6, 10, 15, 21, 28要素
// __constant__ int loop_to_ang[7][28][3] = {
//     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{2, 0, 0}, {1, 1, 0}, {0, 2, 0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 2}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{3, 0, 0}, {2, 1, 0}, {1, 2, 0}, {0, 3, 0}, {2, 0, 1}, {1, 1, 1}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {0, 0, 3}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{4, 0, 0}, {3, 1, 0}, {2, 2, 0}, {1, 3, 0}, {0, 4, 0}, {3, 0, 1}, {2, 1, 1}, {1, 2, 1}, {0, 3, 1}, {2, 0, 2}, {1, 1, 2}, {0, 2, 2}, {1, 0, 3}, {0, 1, 3}, {0, 0, 4}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{5, 0, 0}, {4, 1, 0}, {3, 2, 0}, {2, 3, 0}, {1, 4, 0}, {0, 5, 0}, {4, 0, 1}, {3, 1, 1}, {2, 2, 1}, {1, 3, 1}, {0, 4, 1}, {3, 0, 2}, {2, 1, 2}, {1, 2, 2}, {0, 3, 2}, {2, 0, 3}, {1, 1, 3}, {0, 2, 3}, {1, 0, 4}, {0, 1, 4}, {0, 0, 5}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{6, 0, 0}, {5, 1, 0}, {4, 2, 0}, {3, 3, 0}, {2, 4, 0}, {1, 5, 0}, {0, 6, 0}, {5, 0, 1}, {4, 1, 1}, {3, 2, 1}, {2, 3, 1}, {1, 4, 1}, {0, 5, 1}, {4, 0, 2}, {3, 1, 2}, {2, 2, 2}, {1, 3, 2}, {0, 4, 2}, {3, 0, 3}, {2, 1, 3}, {1, 2, 3}, {0, 3, 3}, {2, 0, 4}, {1, 1, 4}, {0, 2, 4}, {1, 0, 5}, {0, 1, 5}, {0, 0, 6}}
// };

// 順序合わせたやつ
extern __constant__ int loop_to_ang[7][28][3];







//msbを取得
inline __device__ 
int get_msb(int n)
{
  int i = 0;
 
  while (n > 0) {
    n >>= 1;
    i++;
  }
 
  /* return -1 if a == 0. */
  return i - 1;
}

//double型変数の"整数乗"を計算 powより多分速い
inline __device__ 
double right2left_binary_woif(double a, int n)
{
  double x = 1;
  int i = get_msb(n);
  int j = 0;
 
  for (j = 0; j <= i; j++) {
    x *=(((n>>j)&1)*(a-1) + 1);
    a *= a;
  }

  return x;
}


//*
extern __constant__ int tuv_list[2925][3];
//*/






} // namespace gansu::gpu

#endif