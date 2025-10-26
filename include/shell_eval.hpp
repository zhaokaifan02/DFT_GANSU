// shell_eval.hpp
#pragma once
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "types.hpp"     // Atom, Coordinate, AngularMomentums, angstrom_to_bohr
#include "basis_set.hpp" // BasisSet, ElementBasisSet, ContractedGauss
// ====================== AO “函数对象”：一条记录就是一个 AO 分量 ======================
struct AODesc
{
    int atom;                   // 该 AO 所在的原子索引
    int l;                      // 壳角动量
    int lx, ly, lz;             // 笛卡尔幂次数
    std::vector<double> exps;   // α_j
    std::vector<double> coeffs; // c_j（含或不含归一化，视你的设定）
};
static inline void cartesian_triples_for_l(int l, std::vector<std::tuple<int, int, int>> &out)
{
    out.clear();
    out.reserve((l + 1) * (l + 2) / 2);
    for (int lx = l; lx >= 0; --lx)
    {
        for (int ly = l - lx; ly >= 0; --ly)
        {
            int lz = l - lx - ly;
            out.emplace_back(lx, ly, lz);
        }
    }
}

// ============= 可选：笛卡尔 primitive 与收缩归一化（建议开，结果更靠谱） =============
static inline double double_factorial_odd(int n)
{
    if (n <= 0)
        return 1.0;
    double v = 1.0;
    for (int k = n; k >= 1; k -= 2)
        v *= k;
    return v;
}
// primitive 归一化常数（笛卡尔，高斯 × r^l）
static inline double prim_norm_cart(double alpha, int l)
{
    // N = ( 2^(2l+3/2) * alpha^(l+3/2) / (sqrt(pi) * (2l-1)!!) )^(1/2)
    const double df = double_factorial_odd(2 * l - 1);
    const double num = std::pow(2.0, 2 * l + 1.5) * std::pow(alpha, l + 1.5);
    const double den = std::sqrt(M_PI) * df;
    return std::sqrt(num / den);
}
static inline void normalize_contracted_cart(
    int l,
    std::vector<double> &exps,
    std::vector<double> &coeffs)
{
    const int nprim = (int)exps.size();
    if ((int)coeffs.size() != nprim)
        throw std::runtime_error("normalize: size mismatch");
    // 先乘 primitive 归一化，再统一收缩归一化
    double sum2 = 0.0;
    std::vector<double> tmp(coeffs);
    for (int i = 0; i < nprim; ++i)
    {
        for (int j = 0; j < nprim; ++j)
        {
            double Ni = prim_norm_cart(exps[i], l);
            double Nj = prim_norm_cart(exps[j], l);
            double alpha_sum = exps[i] + exps[j];
            // 高斯乘积的重叠积分
            double overlap = Ni * Nj * tmp[i] * tmp[j] *
                             std::pow(M_PI / alpha_sum, 1.5) *
                             std::pow(2.0 / alpha_sum, l);
            sum2 += overlap;
        }
    }

    const double Nc = 1.0 / std::sqrt(std::max(sum2, 1e-300));
    for (int i = 0; i < nprim; ++i)
    {
        double Ni = prim_norm_cart(exps[i], l);
        coeffs[i] = coeffs[i] * Ni * Nc;
    }
}



// 根据 gbs + 分子生成 AO 列表（每个笛卡尔分量算一个 AO）
struct BuildOptions
{
    bool normalize_cart = true; // 建议 true
};

// 角动量字符到 l
static inline int shell_char_to_l(char c)
{
    switch (std::toupper(static_cast<unsigned char>(c)))
    {
    case 'S':
        return 0;
    case 'P':
        return 1;
    case 'D':
        return 2;
    case 'F':
        return 3;
    case 'G':
        return 4;
    case 'H':
        return 5;
    case 'I':
        return 6;
    case 'J':
        return 7;
    default:
        throw std::runtime_error("Unknown shell type");
    }
}

static inline double moment_1d(int n, double a)
{
    // n >= 0
    auto df = double_factorial_odd(2 * n - 1); // 约定 n=0 -> 1
    return std::sqrt(M_PI) * df / (std::pow(2.0, n) * std::pow(a, n + 0.5));
}

// 3D 自重叠：∫ x^{2lx} y^{2ly} z^{2lz} e^{-a r^2} d^3r = Π轴 moment_1d
static inline double cart_self_overlap_xyz(int lx, int ly, int lz, double a)
{
    return moment_1d(lx, a) * moment_1d(ly, a) * moment_1d(lz, a);
}

// 对“当前分量”的收缩系数做单位化（只改 coeff，不改 exponent）
static inline void post_normalize_component_xyz(
    int lx, int ly, int lz,
    const std::vector<double> &exps,
    std::vector<double> &coeffs // inout
)
{
    const int nprim = (int)exps.size();
    double S = 0.0;
    for (int i = 0; i < nprim; ++i)
    {
        for (int j = 0; j < nprim; ++j)
        {
            const double a = exps[i] + exps[j];
            S += coeffs[i] * coeffs[j] * cart_self_overlap_xyz(lx, ly, lz, a);
        }
    }
    const double Nc = 1.0 / std::sqrt(std::max(S, 1e-300));
    for (int i = 0; i < nprim; ++i)
        coeffs[i] *= Nc;
}

// charges: 原子序数 Z；atm_coords: 原子坐标(Bohr)；bs: .gbs 解析结果
static inline std::vector<AODesc>
build_ao_descs_from_mol_and_gbs(
    const std::vector<int> &charges,
    const std::vector<std::array<double, 3>> &atm_coords,
    const gansu::BasisSet &bs,
    const BuildOptions &opt = {})
{
    if (charges.size() != atm_coords.size())
        throw std::runtime_error("charges and atm_coords size mismatch");

    // Z->元素符号：若你的 basis_set.hpp/types.hpp 里已有映射，可改为项目内的函数
    auto Z_to_symbol = [](int Z) -> std::string
    {
        static const char *table[] = {
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"};
        if (Z >= 1 && Z <= (int)(sizeof(table) / sizeof(table[0])))
            return table[Z - 1];
        throw std::runtime_error("Z_to_symbol: unsupported Z");
    };

    std::vector<AODesc> out;
    out.reserve(128);
    for (int A = 0; A < (int)charges.size(); ++A)
    {
        const std::string sym = Z_to_symbol(charges[A]);
        const gansu::ElementBasisSet &ebs = bs.get_element_basis_set(sym);

        const size_t nsh = ebs.get_num_contracted_gausses();
        for (size_t ish = 0; ish < nsh; ++ish)
        {
            const gansu::ContractedGauss &cg = ebs.get_contracted_gauss(ish);
            int l = shell_char_to_l(cg.get_type().at(0));

            // 该壳的 primitive 数据
            std::vector<double> exps;
            exps.reserve(cg.get_num_primitives());
            std::vector<double> coeffs;
            coeffs.reserve(cg.get_num_primitives());
            for (size_t ip = 0; ip < cg.get_num_primitives(); ++ip)
            {
                const auto &pg = cg.get_primitive_gauss(ip);
                exps.push_back(pg.exponent);
                coeffs.push_back(pg.coefficient);
            }
            if (opt.normalize_cart)
                normalize_contracted_cart(l, exps, coeffs);

            // 展开笛卡尔分量 (lx,ly,lz)
            std::vector<std::tuple<int, int, int>> triples;
            cartesian_triples_for_l(l, triples);

            for (auto [lx, ly, lz] : triples)
            {
                std::vector<double> coeffs_xyz = coeffs; // 独立一份
                // 关键：本分量的解析单位化
                post_normalize_component_xyz(lx, ly, lz, exps, coeffs_xyz);

                AODesc ao;
                ao.atom = A;
                ao.l = l;
                ao.lx = lx;
                ao.ly = ly;
                ao.lz = lz;
                ao.exps = exps;
                ao.coeffs = std::move(coeffs_xyz);
                out.emplace_back(std::move(ao));
            }
        }
    }
    return out;
}

// =============== AO 值计算：给网格 (xg,yg,zg) ，评估所有 AO ===============
static inline void evaluate_AO_on_grid(
    const std::vector<AODesc> &aos,                       // nao 个 AO
    const std::vector<std::array<double, 3>> &atm_coords, // 原子位置 (Bohr)
    const std::vector<std::array<double, 3>> &grids,      // 网格坐标列表
    std::vector<double> &ao_out                           // 输出，大小 = nao * ngrids，行主序：ao[mu*G + g]
)
{
    const int nao = (int)aos.size();
    const int ngrids = (int)grids.size();
    ao_out.assign((size_t)nao * ngrids, 0.0);

    for (int mu = 0; mu < nao; ++mu)
    {
        const AODesc &ao = aos[mu];
        const auto &R = atm_coords[ao.atom];

        for (int g = 0; g < ngrids; ++g)
        {
            double rx = grids[g][0] - R[0];
            double ry = grids[g][1] - R[1];
            double rz = grids[g][2] - R[2];
            double r2 = rx * rx + ry * ry + rz * rz;

            // 收缩径向部分 Σ c_i exp(-α_i r^2)
            double radial = 0.0;
            const int nprim = (int)ao.exps.size();
            for (int i = 0; i < nprim; ++i)
            {
                radial += ao.coeffs[i] * std::exp(-ao.exps[i] * r2);
            }

            // 角向笛卡尔多项式 r_x^lx r_y^ly r_z^lz
            // （注意 0 次幂=1；负零幂不会出现）
            double ang = 1.0;
            if (ao.lx)
                ang *= std::pow(rx, ao.lx);
            if (ao.ly)
                ang *= std::pow(ry, ao.ly);
            if (ao.lz)
                ang *= std::pow(rz, ao.lz);

            ao_out[(size_t)mu * ngrids + g] = ang * radial;
        }
    }
}

inline std::vector<uint8_t> build_screen_index_no_omp(
    const std::vector<AODesc> &aos,
    const std::vector<std::array<double, 3>> &atm_coords, // Bohr
    const std::vector<std::array<double, 3>> &grid,       // AoS (G×3)
    int blksize = 56,                                     // 建议 64/128/256
    int nbins   = 100,                                    // 建议 32~128
    double cutoff = 1e-15                                 // >0，且别太离谱
)
{
    // ---- 基本合法性 ----
    if (grid.empty() || aos.empty()) return {};
    if (blksize <= 0) throw std::runtime_error("build_screen_index_no_omp: blksize must be > 0");
    if (!(cutoff > 0.0)) throw std::runtime_error("build_screen_index_no_omp: cutoff must be > 0");

    const int G    = (int)grid.size();
    const int nAO  = (int)aos.size();
    const int nblk = (G + blksize - 1) / blksize;

    nbins = std::min(std::max(nbins, 1), 127);

    // 防止 log(<=0)
    const double cutoff_c = std::min(cutoff, 0.1);
    if (!(cutoff_c > 0.0)) throw std::runtime_error("build_screen_index_no_omp: cutoff too small");

    const double scale = - (double)nbins / std::log(cutoff_c);

    std::vector<uint8_t> screen_index((size_t)nblk * (size_t)nAO, 0);
    std::vector<double>  rr_buf((size_t)blksize);

    for (int ao_id = 0; ao_id < nAO; ++ao_id) {
        const AODesc &ao = aos[ao_id];
        const int l = ao.l;

        // 原子下标与数据合法性
        if (ao.atom < 0 || ao.atom >= (int)atm_coords.size()) continue;             // 整列 0
        if (ao.exps.empty() || ao.coeffs.empty())            continue;              // 整列 0

        // 可选：若你要求严格等长，就启用下面一行
        // if (ao.exps.size() != ao.coeffs.size()) continue;

        const double ax = atm_coords[ao.atom][0];
        const double ay = atm_coords[ao.atom][1];
        const double az = atm_coords[ao.atom][2];

        double min_exp = 1e300;
        for (double a : ao.exps) min_exp = std::min(min_exp, a);

        double maxc = 0.0;
        for (double c : ao.coeffs) maxc = std::max(maxc, std::fabs(c));

        if (!(min_exp > 0.0) || !(maxc > 0.0)) continue;    // 避免 log(0)、负指数

        const double log_coeff = std::log(maxc);

        // l>0 的内区界
        double r2sup    = 0.0;
        double a_rr_min = -log_coeff;
        if (l > 0) {
            r2sup    = (double)l / (2.0 * min_exp);
            if (r2sup <= 0.0 || !std::isfinite(r2sup)) continue;
            a_rr_min = min_exp * r2sup - 0.5 * std::log(r2sup) * l - log_coeff;
        }

        for (int ib = 0; ib < nblk; ++ib) {
            const int g0 = ib * blksize;
            const int g1 = std::min(G, (ib + 1) * blksize);
            const int dg = g1 - g0;
            if (dg <= 0) continue;

            for (int i = 0; i < dg; ++i) {
                const double dx = grid[g0 + i][0] - ax;
                const double dy = grid[g0 + i][1] - ay;
                const double dz = grid[g0 + i][2] - az;
                rr_buf[(size_t)i] = dx*dx + dy*dy + dz*dz;
            }

            double rr_min = 1e300;
            for (int i = 0; i < dg; ++i) rr_min = std::min(rr_min, rr_buf[(size_t)i]);

            double a_rr;
            if (l == 0) {
                a_rr = min_exp * rr_min - log_coeff;
            } else if (rr_min < r2sup) {
                a_rr = a_rr_min;                          // 避免 log(0)
            } else {
                // rr_min > 0 时才会走到这里：安全
                a_rr = min_exp * rr_min - 0.5 * std::log(rr_min) * l - log_coeff;
            }

            // 映射为离散等级（鲁棒处理 NaN/Inf）
            uint8_t si_byte = 0;
            if (std::isfinite(a_rr)) {
                const double si_real = (double)nbins - a_rr * scale;
                if (si_real > 0.0 && std::isfinite(si_real)) {
                    const int si = (int)(si_real + 1.0);
                    si_byte = (uint8_t)std::min(si, nbins);
                }
            }
            screen_index[(size_t)ib * (size_t)nAO + (size_t)ao_id] = si_byte;
        }
    }
    return screen_index;
}

void evaluate_AO_on_grid_screened(
    const std::vector<AODesc> &aos,                       // nao
    const std::vector<std::array<double, 3>> &atm_coords, // A×3 (Bohr)
    const std::vector<std::array<double, 3>> &grids,      // G×3 (AoS)
    int blksize,                                          // 与构建 screen_index 时一致
    const std::vector<uint8_t> &screen_index,             // 大小: nblk*nao
    std::vector<double> &ao_out                           // 输出: nao*G
);


#ifdef DEMO_MAIN
int main()
{
    // 1) 分子（例如 H2O；Bohr）
    std::vector<int> charges = {8, 1, 1};
    std::vector<std::array<double, 3>> atm_coords = {
        {0.0, 0.0, 0.0},
        {0.0, 1.432336, 1.107152},
        {0.0, -1.432336, 1.107152}};

    // 2) 读取基组
    gansu::BasisSet bs;
    bs.construct_from_gbs("sto-3g.gbs");

    // 3) 展开 AO 描述（不需要 atm/bas/env）
    auto aos = build_ao_descs_from_mol_and_gbs(charges, atm_coords, bs, {/*normalize_cart*/ true});
    int nao = (int)aos.size();

    // 4) 构造一些网格点（示例：2x2x2 立方 8 点）
    std::vector<std::array<double, 3>> grids;
    for (int iz = 0; iz < 2; ++iz)
        for (int iy = 0; iy < 2; ++iy)
            for (int ix = 0; ix < 2; ++ix)
            {
                grids.push_back({-0.5 + ix * 1.0, -0.5 + iy * 1.0, -0.5 + iz * 1.0});
            }
    int G = (int)grids.size();

    // 5) 评估 AO
    std::vector<double> AO; // 大小 = nao * G
    evaluate_AO_on_grid(aos, atm_coords, grids, AO);

    // 6) 打印前几项检查
    std::printf("nao=%d, ngrids=%d\n", nao, G);
    for (int mu = 0; mu < std::min(nao, 6); ++mu)
    {
        std::printf("AO[%d]:", mu);
        for (int g = 0; g < std::min(G, 4); ++g)
            std::printf(" %.6e", AO[(size_t)mu * G + g]);
        std::printf("\n");
    }
    return 0;
}
#endif
