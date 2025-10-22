// // shell_eval.cpp
#include "shell_eval.hpp"
// void evaluate_AO_on_grid_screened(
//     const std::vector<AODesc> &aos,                       // nao
//     const std::vector<std::array<double, 3>> &atm_coords, // A×3 (Bohr)
//     const std::vector<std::array<double, 3>> &grids,      // G×3 (AoS)
//     int blksize,                                          // 与构建 screen_index 时一致
//     const std::vector<uint8_t> &screen_index,             // 大小: nblk*nao
//     std::vector<double> &ao_out                           // 输出: nao*G
// )
// {
//     const int nao = (int)aos.size();
//     const int G = (int)grids.size();
//     const int nblk = (G + blksize - 1) / blksize;

//     if ((int)screen_index.size() != nblk * nao)
//     {
//         throw std::runtime_error("evaluate_AO_on_grid_screened: screen_index size mismatch");
//     }
//     ao_out.assign((size_t)nao * G, 0.0);

//     for (int ib = 0; ib < nblk; ++ib)
//     {
//         const int g0 = ib * blksize;
//         const int g1 = std::min(G, (ib + 1) * blksize);
//         const int dg = g1 - g0;

//         // 遍历 AO；若该 block 的该 AO 被屏蔽（=0）则整块跳过
//         for (int mu = 0; mu < nao; ++mu)
//         {
//             if (screen_index[ib * nao + mu] == 0)
//                 continue;
// 
//             const AODesc &ao = aos[mu];
//             const auto &R = atm_coords[ao.atom];
//             const int nprim = (int)ao.exps.size();

//             // 逐点评估该 AO（只在这个 block 内）
//             for (int i = 0; i < dg; ++i)
//             {
//                 const int g = g0 + i;

//                 const double rx = grids[g][0] - R[0];
//                 const double ry = grids[g][1] - R[1];
//                 const double rz = grids[g][2] - R[2];
//                 const double r2 = rx * rx + ry * ry + rz * rz;

//                 // 收缩径向部分 Σ c_i exp(-α_i r^2)
//                 double radial = 0.0;
//                 // 手写展开/向量化可再加速；这里保持清爽
//                 for (int p = 0; p < nprim; ++p)
//                 {
//                     radial += ao.coeffs[p] * std::exp(-ao.exps[p] * r2);
//                 }

//                 // 角向笛卡尔多项式 r_x^lx r_y^ly r_z^lz
//                 double ang = 1.0;
//                 if (ao.lx)
//                     ang *= std::pow(rx, ao.lx);
//                 if (ao.ly)
//                     ang *= std::pow(ry, ao.ly);
//                 if (ao.lz)
//                     ang *= std::pow(rz, ao.lz);

//                 ao_out[(size_t)mu * G + g] = ang * radial;
//             }
//         }
//     }
// }


void evaluate_AO_on_grid_screened(
    const std::vector<AODesc> &aos,
    const std::vector<std::array<double, 3>> &atm_coords,
    const std::vector<std::array<double, 3>> &grids,
    int blksize,
    const std::vector<uint8_t> &screen_index,
    std::vector<double> &ao_out
)
{
    const int nao = (int)aos.size();
    const int G = (int)grids.size();
    const int nblk = (G + blksize - 1) / blksize;

    if ((int)screen_index.size() != nblk * nao)
        throw std::runtime_error("evaluate_AO_on_grid_screened: screen_index size mismatch");

    ao_out.assign((size_t)nao * G, 0.0);

    // ---------- 跳过统计 ----------
    size_t total_blocks = (size_t)nblk * nao;
    size_t skipped_blocks = 0;
    size_t skipped_points = 0;

    for (int ib = 0; ib < nblk; ++ib)
    {
        const int g0 = ib * blksize;
        const int g1 = std::min(G, (ib + 1) * blksize);
        const int dg = g1 - g0;

        for (int mu = 0; mu < nao; ++mu)
        {
            if (screen_index[ib * nao + mu] == 0)
            {
                continue;
            }

            const AODesc &ao = aos[mu];
            const auto &R = atm_coords[ao.atom];
            const int nprim = (int)ao.exps.size();

            for (int i = 0; i < dg; ++i)
            {
                const int g = g0 + i;
                const double rx = grids[g][0] - R[0];
                const double ry = grids[g][1] - R[1];
                const double rz = grids[g][2] - R[2];
                const double r2 = rx * rx + ry * ry + rz * rz;

                double radial = 0.0;
                for (int p = 0; p < nprim; ++p)
                    radial += ao.coeffs[p] * std::exp(-ao.exps[p] * r2);

                double ang = 1.0;
                if (ao.lx) ang *= std::pow(rx, ao.lx);
                if (ao.ly) ang *= std::pow(ry, ao.ly);
                if (ao.lz) ang *= std::pow(rz, ao.lz);

                ao_out[(size_t)mu * G + g] = ang * radial;
            }
        }
    }

    // ---------- 打印统计结果 ----------
    double ratio_blocks = 100.0 * skipped_blocks / (double)total_blocks;
    double ratio_points = 100.0 * skipped_points / (double)(G * nao);

    printf("[screened AO eval] skipped blocks: %zu / %zu (%.2f%%)\n",
           skipped_blocks, total_blocks, ratio_blocks);
    printf("[screened AO eval] approx skipped points: %zu / %zu (%.2f%%)\n",
           skipped_points, (size_t)G * nao, ratio_points);
}

// std::vector<ShellEval>
// instantiate_for_molecule(const gansu::BasisSet &bs,
//                          const std::vector<gansu::Atom> &atoms,
//                          bool normalize_primitives,
//                          double coeff_prune_tol)
// {
//     std::vector<ShellEval> shells;
//     shells.reserve(64);

//     for (size_t ia = 0; ia < atoms.size(); ++ia)
//     {
//         const auto &A = atoms[ia];
//         const std::string elem = gansu::atomic_number_to_element_name(A.atomic_number);
//         const auto &ebs = bs.get_element_basis_set(elem); // 需要你提供访问器

//         for (const auto &cg : ebs.get_contracted_gaussians())
//         { // 访问器
//             ShellEval sh;
//             sh.l = gansu::shell_name_to_shell_type(cg.get_type()); // "S","P",...
//             sh.center = {gansu::angstrom_to_bohr(A.coordinate.x),
//                          gansu::angstrom_to_bohr(A.coordinate.y),
//                          gansu::angstrom_to_bohr(A.coordinate.z)};
//             sh.atom_index = static_cast<int>(ia);

//             // 收集 primitive
//             const auto &prims = cg.get_primitives(); // 返回 {exponent, coefficient} 列表
//             sh.nprim = static_cast<int>(prims.size());
//             sh.nctr = 1; // 一般收缩>1时可扩展
//             sh.alpha.resize(sh.nprim);
//             sh.coeff.assign(sh.nctr, std::vector<double>(sh.nprim, 0.0));

//             for (int p = 0; p < sh.nprim; ++p)
//             {
//                 sh.alpha[p] = prims[p].exponent;
//                 double c = prims[p].coefficient;
//                 if (normalize_primitives)
//                     c *= gto_norm_s_like(sh.l, sh.alpha[p]);
//                 sh.coeff[0][p] = c;
//             }

//             // 按 alpha 降序 + 裁剪
//             std::vector<int> order(sh.nprim);
//             std::iota(order.begin(), order.end(), 0);
//             std::sort(order.begin(), order.end(),
//                       [&](int i, int j)
//                       { return sh.alpha[i] > sh.alpha[j]; });
//             std::vector<double> a2(sh.nprim), c2(sh.nprim);
//             for (int k = 0; k < sh.nprim; ++k)
//             {
//                 a2[k] = sh.alpha[order[k]];
//                 c2[k] = sh.coeff[0][order[k]];
//             }
//             int kept = 0;
//             for (int k = 0; k < sh.nprim; ++k)
//                 if (std::abs(c2[k]) > coeff_prune_tol)
//                 {
//                     a2[kept] = a2[k];
//                     c2[kept] = c2[k];
//                     ++kept;
//                 }
//             a2.resize(kept);
//             c2.resize(kept);
//             sh.alpha = std::move(a2);
//             sh.coeff[0] = std::move(c2);
//             sh.nprim = static_cast<int>(sh.alpha.size());

//             // 笛卡尔分量列表
//             sh.cart_xyz.clear();
//             for (const auto &xyz : gansu::AngularMomentums[sh.l])
//             {
//                 sh.cart_xyz.push_back({xyz[0], xyz[1], xyz[2]});
//             }
//             shells.emplace_back(std::move(sh));
//         }
//     }

//     // 计算全局 ao_base
//     size_t base = 0;
//     for (auto &sh : shells)
//     {
//         sh.ao_base = base;
//         base += sh.cart_xyz.size() * sh.nctr; // nctr>1 时乘列数
//     }
//     return shells;
// }

// void eval_ao_cart_cpu(const std::vector<ShellEval> &shells,
//                       const double *gx, const double *gy, const double *gz,
//                       int ngrid, double *ao)
// {
//     for (const auto &sh : shells)
//     {
//         const double XA = sh.center.x, YA = sh.center.y, ZA = sh.center.z;

//         // 预取平移（可选）
//         std::vector<double> dx(ngrid), dy(ngrid), dz(ngrid), r2(ngrid);
//         for (int g = 0; g < ngrid; ++g)
//         {
//             dx[g] = gx[g] - XA;
//             dy[g] = gy[g] - YA;
//             dz[g] = gz[g] - ZA;
//             r2[g] = dx[g] * dx[g] + dy[g] * dy[g] + dz[g] * dz[g];
//         }

//         const size_t base = sh.ao_base;
//         const int ncomp = (int)sh.cart_xyz.size();

//         for (int c = 0; c < ncomp; ++c)
//         {
//             const int lx = sh.cart_xyz[c][0], ly = sh.cart_xyz[c][1], lz = sh.cart_xyz[c][2];
//             const size_t ao_row = base + c;

//             // 逐格点
//             for (int g = 0; g < ngrid; ++g)
//             {
//                 // 多项式
//                 double px = 1.0, py = 1.0, pz = 1.0;
//                 for (int i = 0; i < lx; ++i)
//                     px *= dx[g];
//                 for (int i = 0; i < ly; ++i)
//                     py *= dy[g];
//                 for (int i = 0; i < lz; ++i)
//                     pz *= dz[g];

//                 // primitive 收缩（含该分量的规范化）
//                 double S = 0.0;
//                 for (int p = 0; p < sh.nprim; ++p)
//                 {
//                     const double a = sh.alpha[p];
//                     const double nrm = norm_cart_primitive(lx, ly, lz, a);
//                     S += (sh.coeff[0][p] * nrm) * std::exp(-a * r2[g]);
//                 }

//                 ao[ao_row * ngrid + g] = S * px * py * pz;
//             }
//         }
//     }
// }
// // 参考pyscf的
// #define ANG_MAX 15 // l = 0..15
// #define LMAX ANG_MAX
// #define SIMDD 8
// // 128s42p21d12f8g6h4i3j
// #define NCTR_CART 128
// #define NPRIMAX 40
// #define BLKSIZE 56
// #define EXPCUTOFF 50 // 1e-22
// #define NOTZERO(e) (fabs(e) > 1e-18)
// #define ALIGN8_UP(buf) (double *)(void *)(((uintptr_t)buf + 7) & (-(uintptr_t)8))
// #define TENSOR 1
// // slots of atm
// #define CHARGE_OF 0
// #define PTR_COORD 1
// #define NUC_MOD_OF 2
// #define PTR_ZETA 3
// #define PTR_FRAC_CHARGE 4
// #define RESERVE_ATMSLOT 5
// #define ATM_SLOTS 6

// // slots of bas
// #define ATOM_OF 0
// #define ANG_OF 1
// #define NPRIM_OF 2
// #define NCTR_OF 3
// #define KAPPA_OF 4
// #define PTR_EXP 5
// #define PTR_COEFF 6
// #define RESERVE_BASLOT 7
// #define BAS_SLOTS 8

// // slots of gout
// #define POSX 0
// #define POSY 1
// #define POSZ 2
// #define POS1 3

// double CINTcommon_fac_sp(int l)
// {
//     if (l == 0)
//         return 0.28125; // ≈ 9/32
//     else if (l == 1)
//         return 0.49033; // ≈ 1/√(4.16)
//     else
//         return 1.0;
// }
// typedef int (*FPtr_exp)(double *ectr, double *coord, double *alpha, double *coeff,
//                         int l, int nprim, int nctr, size_t ngrids, double fac);
// typedef void (*FPtr_eval)(double *gto, double *ri, double *exps,
//                           double *coord, double *alpha, double *coeff,
//                           double *env, int l, int np, int nc,
//                           size_t nao, size_t ngrids, size_t blksize);
// void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
//                              double *coord, double *alpha, double *coeff,
//                              double *env, int l, int np, int nc,
//                              size_t nao, size_t ngrids, size_t blksize)
// {
//     int lx, ly, lz;
//     size_t i, k;
//     double buf[(LMAX + 1) * 3 * BLKSIZE + 8];
//     double *xpows = ALIGN8_UP(buf);
//     double *ypows = xpows + (LMAX + 1) * BLKSIZE;
//     double *zpows = ypows + (LMAX + 1) * BLKSIZE;
//     double *gridx = coord;
//     double *gridy = coord + BLKSIZE;
//     double *gridz = coord + BLKSIZE * 2;

//     switch (l)
//     {
//     case 0:
//         for (k = 0; k < nc; k++)
//         {
//             for (i = 0; i < blksize; i++)
//             {
//                 gto[k * ngrids + i] = exps[k * BLKSIZE + i];
//             }
//         }
//         break;
//     case 1:
//         for (k = 0; k < nc; k++)
//         {
// #pragma GCC ivdep
//             for (i = 0; i < blksize; i++)
//             {
//                 gto[i] = gridx[i] * exps[k * BLKSIZE + i];
//                 gto[1 * ngrids + i] = gridy[i] * exps[k * BLKSIZE + i];
//                 gto[2 * ngrids + i] = gridz[i] * exps[k * BLKSIZE + i];
//             }
//             gto += ngrids * 3;
//         }
//         break;
//     case 2:
//         for (k = 0; k < nc; k++)
//         {
// #pragma GCC ivdep
//             for (i = 0; i < blksize; i++)
//             {
//                 gto[i] = exps[k * BLKSIZE + i] * gridx[i] * gridx[i];              // xx
//                 gto[1 * ngrids + i] = exps[k * BLKSIZE + i] * gridx[i] * gridy[i]; // xy
//                 gto[2 * ngrids + i] = exps[k * BLKSIZE + i] * gridx[i] * gridz[i]; // xz
//                 gto[3 * ngrids + i] = exps[k * BLKSIZE + i] * gridy[i] * gridy[i]; // yy
//                 gto[4 * ngrids + i] = exps[k * BLKSIZE + i] * gridy[i] * gridz[i]; // yz
//                 gto[5 * ngrids + i] = exps[k * BLKSIZE + i] * gridz[i] * gridz[i]; // zz
//             }
//             gto += ngrids * 6;
//         }
//         break;
//     case 3:
//         for (k = 0; k < nc; k++)
//         {
// #pragma GCC ivdep
//             for (i = 0; i < blksize; i++)
//             {
//                 gto[i] = exps[k * BLKSIZE + i] * gridx[i] * gridx[i] * gridx[i];              // xxx
//                 gto[1 * ngrids + i] = exps[k * BLKSIZE + i] * gridx[i] * gridx[i] * gridy[i]; // xxy
//                 gto[2 * ngrids + i] = exps[k * BLKSIZE + i] * gridx[i] * gridx[i] * gridz[i]; // xxz
//                 gto[3 * ngrids + i] = exps[k * BLKSIZE + i] * gridx[i] * gridy[i] * gridy[i]; // xyy
//                 gto[4 * ngrids + i] = exps[k * BLKSIZE + i] * gridx[i] * gridy[i] * gridz[i]; // xyz
//                 gto[5 * ngrids + i] = exps[k * BLKSIZE + i] * gridx[i] * gridz[i] * gridz[i]; // xzz
//                 gto[6 * ngrids + i] = exps[k * BLKSIZE + i] * gridy[i] * gridy[i] * gridy[i]; // yyy
//                 gto[7 * ngrids + i] = exps[k * BLKSIZE + i] * gridy[i] * gridy[i] * gridz[i]; // yyz
//                 gto[8 * ngrids + i] = exps[k * BLKSIZE + i] * gridy[i] * gridz[i] * gridz[i]; // yzz
//                 gto[9 * ngrids + i] = exps[k * BLKSIZE + i] * gridz[i] * gridz[i] * gridz[i]; // zzz
//             }
//             gto += ngrids * 10;
//         }
//         break;
//     default:
//         for (k = 0; k < nc; k++)
//         {
//             for (i = 0; i < blksize; i++)
//             {
//                 xpows[i] = 1;
//                 ypows[i] = 1;
//                 zpows[i] = 1;
//             }
//             for (lx = 1; lx <= l; lx++)
//             {
// #pragma GCC ivdep
//                 for (i = 0; i < blksize; i++)
//                 {
//                     xpows[lx * BLKSIZE + i] = xpows[(lx - 1) * BLKSIZE + i] * gridx[i];
//                     ypows[lx * BLKSIZE + i] = ypows[(lx - 1) * BLKSIZE + i] * gridy[i];
//                     zpows[lx * BLKSIZE + i] = zpows[(lx - 1) * BLKSIZE + i] * gridz[i];
//                 }
//             }
//             for (lx = l; lx >= 0; lx--)
//             {
//                 for (ly = l - lx; ly >= 0; ly--)
//                 {
//                     lz = l - lx - ly;
// #pragma GCC ivdep
//                     for (i = 0; i < blksize; i++)
//                     {
//                         gto[i] = xpows[lx * BLKSIZE + i] * ypows[ly * BLKSIZE + i] * zpows[lz * BLKSIZE + i] * exps[k * BLKSIZE + i];
//                     }
//                     gto += ngrids;
//                 }
//             }
//         }
//     }
// }
// int GTOcontract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
//                      int l, int nprim, int nctr, size_t ngrids, double fac)
// {
//     size_t i, j, k;
//     double arr, eprim;
//     double rr[BLKSIZE];
//     double *gridx = coord;
//     double *gridy = coord + BLKSIZE;
//     double *gridz = coord + BLKSIZE * 2;

// #pragma GCC ivdep
//     for (i = 0; i < ngrids; i++)
//     {
//         rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
//     }

//     for (i = 0; i < nctr * BLKSIZE; i++)
//     {
//         ectr[i] = 0;
//     }
//     for (j = 0; j < nprim; j++)
//     {
//         for (i = 0; i < ngrids; i++)
//         {
//             arr = alpha[j] * rr[i];
//             eprim = exp(-arr) * fac;
//             for (k = 0; k < nctr; k++)
//             {
//                 ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
//             }
//         }
//     }
//     return 1;
// }
// void _fill_grid2atm(double *grid2atm, double *coord, size_t bgrids, size_t ngrids,
//                     int *atm, int natm, int *bas, int nbas, double *env)
// {
//     int atm_id;
//     size_t ig;
//     double *r_atm;
//     for (atm_id = 0; atm_id < natm; atm_id++)
//     {
//         r_atm = env + atm[PTR_COORD + atm_id * ATM_SLOTS];
// #pragma GCC ivdep
//         for (ig = 0; ig < bgrids; ig++)
//         {
//             grid2atm[0 * BLKSIZE + ig] = coord[0 * ngrids + ig] - r_atm[0];
//             grid2atm[1 * BLKSIZE + ig] = coord[1 * ngrids + ig] - r_atm[1];
//             grid2atm[2 * BLKSIZE + ig] = coord[2 * ngrids + ig] - r_atm[2];
//         }
//         grid2atm += 3 * BLKSIZE;
//     }
// }
// void _dset0(double *out, size_t odim, size_t bgrids, int counts)
// {
//     size_t i, j;
//     for (i = 0; i < counts; i++)
//     {
//         for (j = 0; j < bgrids; j++)
//         {
//             out[i * odim + j] = 0;
//         }
//     }
// }

// void GTOeval_cart_iter(FPtr_eval feval, FPtr_exp fexp, double fac,
//                        size_t nao, size_t ngrids, size_t bgrids,
//                        int param[], int *shls_slice, int *ao_loc, double *buf,
//                        double *ao, double *coord, uint8_t *non0table,
//                        int *atm, int natm, int *bas, int nbas, double *env)
// {
//     const int ncomp = param[TENSOR];
//     const int sh0 = shls_slice[0];
//     const int sh1 = shls_slice[1];
//     const int atmstart = bas[sh0 * BAS_SLOTS + ATOM_OF];
//     const int atmend = bas[(sh1 - 1) * BAS_SLOTS + ATOM_OF] + 1;
//     const int atmcount = atmend - atmstart;
//     int i, l, np, nc, atm_id, bas_id, deg, ao_id;
//     double fac1;
//     double *p_exp, *pcoeff, *pcoord, *ri;
//     double *grid2atm = ALIGN8_UP(buf); // [atm_id,xyz,grid]
//     double *eprim = grid2atm + atmcount * 3 * BLKSIZE;

//     _fill_grid2atm(grid2atm, coord, bgrids, ngrids,
//                    atm + atmstart * ATM_SLOTS, atmcount, bas, nbas, env);

//     for (bas_id = sh0; bas_id < sh1; bas_id++)
//     {
//         np = bas[bas_id * BAS_SLOTS + NPRIM_OF];
//         nc = bas[bas_id * BAS_SLOTS + NCTR_OF];
//         l = bas[bas_id * BAS_SLOTS + ANG_OF];
//         deg = (l + 1) * (l + 2) / 2;
//         fac1 = fac * CINTcommon_fac_sp(l);
//         p_exp = env + bas[bas_id * BAS_SLOTS + PTR_EXP];
//         pcoeff = env + bas[bas_id * BAS_SLOTS + PTR_COEFF];
//         atm_id = bas[bas_id * BAS_SLOTS + ATOM_OF];
//         pcoord = grid2atm + (atm_id - atmstart) * 3 * BLKSIZE;
//         ao_id = ao_loc[bas_id] - ao_loc[sh0];
//         if (non0table[bas_id] &&
//             (*fexp)(eprim, pcoord, p_exp, pcoeff, l, np, nc, bgrids, fac1))
//         {
//             ri = env + atm[PTR_COORD + atm_id * ATM_SLOTS];
//             (*feval)(ao + ao_id * ngrids, ri, eprim, pcoord, p_exp, pcoeff,
//                      env, l, np, nc, nao, ngrids, bgrids);
//         }
//         else
//         {
//             for (i = 0; i < ncomp; i++)
//             {
//                 _dset0(ao + (i * nao + ao_id) * ngrids, ngrids, bgrids, nc * deg);
//             }
//         }
//     }
// }
// int GTOshloc_by_atom(int *shloc, int *shls_slice, int *ao_loc, int *atm, int *bas)
// {
//     const int sh0 = shls_slice[0];
//     const int sh1 = shls_slice[1];
//     int ish, nshblk, lastatm;
//     shloc[0] = sh0;
//     nshblk = 1;
//     lastatm = bas[BAS_SLOTS * sh0 + ATOM_OF];
//     for (ish = sh0; ish < sh1; ish++)
//     {
//         if (lastatm != bas[BAS_SLOTS * ish + ATOM_OF])
//         {
//             lastatm = bas[BAS_SLOTS * ish + ATOM_OF];
//             shloc[nshblk] = ish;
//             nshblk++;
//         }
//     }
//     shloc[nshblk] = sh1;
//     return nshblk;
// }
// #define POS_E1 0
// #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
// void GTOeval_loop(void (*fiter)(FPtr_eval feval, FPtr_exp fexp, double fac,
//                                 size_t nao, size_t ngrids, size_t bgrids,
//                                 int param[], int *shls_slice, int *ao_loc, double *buf,
//                                 double *ao, double *coord, uint8_t *non0table,
//                                 int *atm, int natm, int *bas, int nbas, double *env),
//                   FPtr_eval feval, FPtr_exp fexp, double fac,
//                   int ngrids, int param[], int *shls_slice, int *ao_loc,
//                   double *ao, double *coord, uint8_t *non0table,
//                   int *atm, int natm, int *bas, int nbas, double *env)
// {
//     int shloc[shls_slice[1] - shls_slice[0] + 1];
//     const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
//     const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
//     const size_t Ngrids = ngrids;

// #pragma omp parallel
//     {
//         const int sh0 = shls_slice[0];
//         const int sh1 = shls_slice[1];
//         const size_t nao = ao_loc[sh1] - ao_loc[sh0];
//         int ip, ib, k, iloc, ish;
//         size_t aoff, bgrids;
//         int ncart = NCTR_CART * param[TENSOR] * param[POS_E1];
//         double *buf = (double *)malloc(sizeof(double) * BLKSIZE * (NPRIMAX * 2 + ncart + 1));
// #pragma omp for schedule(dynamic, 4)
//         for (k = 0; k < nblk * nshblk; k++)
//         {
//             iloc = k / nblk;
//             ish = shloc[iloc];
//             aoff = ao_loc[ish] - ao_loc[sh0];
//             ib = k - iloc * nblk;
//             ip = ib * BLKSIZE;
//             bgrids = MIN(ngrids - ip, BLKSIZE);
//             (*fiter)(feval, fexp, fac, nao, Ngrids, bgrids,
//                      param, shloc + iloc, ao_loc, buf, ao + aoff * Ngrids + ip,
//                      coord + ip, non0table + ib * nbas,
//                      atm, natm, bas, nbas, env);
//         }
//         free(buf);
//     }
// }

// void GTOeval_cart_drv(FPtr_eval feval, FPtr_exp fexp, double fac, int ngrids,
//                       int param[], int *shls_slice, int *ao_loc,
//                       double *ao, double *coord, uint8_t *non0table,
//                       int *atm, int natm, int *bas, int nbas, double *env)
// {
//     GTOeval_loop(GTOeval_cart_iter, feval, fexp, fac, ngrids,
//                  param, shls_slice, ao_loc,
//                  ao, coord, non0table, atm, natm, bas, nbas, env);
// }

// void GTOval_cart(int ngrids, int *shls_slice, int *ao_loc,
//                  double *ao, double *coord, uint8_t *non0table,
//                  int *atm, int natm, int *bas, int nbas, double *env)
// {
//     int param[] = {1, 1};
//     GTOeval_cart_drv(GTOshell_eval_grid_cart, GTOcontract_exp0, 1,
//                      ngrids, param, shls_slice, ao_loc,
//                      ao, coord, non0table, atm, natm, bas, nbas, env);
// }

// void GTOval_cart_deriv0(int ngrids, int *shls_slice, int *ao_loc,
//                         double *ao, double *coord, uint8_t *non0table,
//                         int *atm, int natm, int *bas, int nbas, double *env)
// {
//     GTOval_cart(ngrids, shls_slice, ao_loc,
//                 ao, coord, non0table, atm, natm, bas, nbas, env);
// }



