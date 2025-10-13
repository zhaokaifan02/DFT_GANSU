#include "dft_atomGrid.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <array>
#include <stdexcept>
namespace cg = chemgrid::radial;

void cg::delley(int n, std::vector<double> &r, std::vector<double> &dr)
{
    r.resize(n);
    dr.resize(n);
    const double r_outer = 12.0;
    const double step = 1.0 / (n + 1);
    const double t = n * step;                     // n/(n+1)
    const double denom = 1.0 - t * t;              // > 0
    const double rfac = r_outer / std::log(denom); // < 0

    for (int i = 1; i <= n; ++i)
    {
        const double s = i * step;
        const double val = 1.0 - s * s;
        r[i - 1] = rfac * std::log(val);
        dr[i - 1] = rfac * (-2.0 * i * (step * step)) / val;
    }
}

void cg::mura_knowles(int n, int charge, std::vector<double> &r, std::vector<double> &dr)
{
    r.resize(n);
    dr.resize(n);
    const double far = (charge == 3 || charge == 4 || charge == 11 || charge == 12 || charge == 19 || charge == 20) ? 7.0 : 5.2;
    const double invn = 1.0 / n;

    for (int i = 0; i < n; ++i)
    {
        const double x = (i + 0.5) * invn;
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double one_minus_x3 = 1.0 - x3;
        r[i] = -far * std::log(one_minus_x3);
        dr[i] = far * 3.0 * x2 / (one_minus_x3 * n);
    }
}

void cg::gauss_chebyshev(int n, std::vector<double> &r, std::vector<double> &dr)
{
    r.resize(n);
    dr.resize(n);
    const double ln2 = 1.0 / std::log(2.0);
    const double fac = 16.0 / 3.0 / (n + 1);
    const double inv_np1 = 1.0 / (n + 1);
    const double pi = 3.141592653589793238462643383279502884;

    std::vector<double> xi_tmp(n), sin1(n), xi(n);

    for (int k = 0; k < n; ++k)
    {
        const double x1 = (k + 1) * pi * inv_np1;
        const double s = std::sin(x1);
        sin1[k] = s;
        const double termA = ((n - 1) - 2.0 * k) * inv_np1;
        const double termB = (1.0 + (2.0 / 3.0) * (s * s)) * std::sin(2.0 * x1) / pi;
        xi_tmp[k] = termA + termB;
    }
    // 反对称化
    for (int k = 0; k < n; ++k)
    {
        xi[k] = 0.5 * (xi_tmp[k] - xi_tmp[n - 1 - k]);
    }

    for (int k = 0; k < n; ++k)
    {
        r[k] = 1.0 - std::log(1.0 + xi[k]) * ln2;
        const double s = sin1[k];
        dr[k] = fac * (s * s) * (s * s) * ln2 / (1.0 + xi[k]);
    }
}

void cg::treutler_ahlrichs(int n, int Z, bool atom_specific,
                           const double *xi_table, std::size_t xi_size,
                           std::vector<double> &r, std::vector<double> &dr)
{
    r.resize(n);
    dr.resize(n);
    double xi = 1.0;
    if (atom_specific && xi_table && xi_size > 0)
    {
        if (Z >= 0 && static_cast<std::size_t>(Z) < xi_size)
            xi = xi_table[Z];
    }
    const double step = 3.141592653589793238462643383279502884 / (n + 1);
    const double ln2 = xi / std::log(2.0);

    // 原函数返回 r/dr 的反序，我们这里直接反向写入
    for (int i = 0; i < n; ++i)
    {
        const double t = (i + 1) * step;
        const double x = std::cos(t);
        const double one_plus_x = 1.0 + x;
        const double one_minus_x = 1.0 - x;
        const double logterm = std::log(one_minus_x / 2.0);
        const double base = ln2 * std::pow(one_plus_x, 0.6);
        const double rval = -base * logterm;
        const double drval = step * std::sin(t) * base * (-0.6 / one_plus_x * logterm + 1.0 / one_minus_x);
        const int j = n - 1 - i;
        r[j] = rval;
        dr[j] = drval;
    }
}

namespace cg1 = chemgrid::prune;

namespace
{
    constexpr std::array<std::array<double, 4>, 3> SG1_ALPHAS{{
        {{0.25, 0.5, 1.0, 4.5}},   // nuc <= 2
        {{0.1667, 0.5, 0.9, 3.5}}, // 3..10
        {{0.1, 0.4, 0.8, 2.5}},    // >10
    }};
    constexpr std::array<int, 5> SG1_LEB_SIZES{{6, 38, 86, 194, 86}};

    inline const std::array<double, 4> &pick_alpha_row(int nuc)
    {
        if (nuc <= 2)
            return SG1_ALPHAS[0];
        if (nuc <= 10)
            return SG1_ALPHAS[1];
        return SG1_ALPHAS[2];
    }
} // anon

std::vector<int> cg1::sg1(int nuc,
                          const std::vector<double> &rads,
                          int /*n_ang*/,
                          const double *radii)
{
    std::vector<int> out;
    out.reserve(rads.size());
    const auto &alpha = pick_alpha_row(nuc);
    const double r_atom = radii[nuc] + 1e-200;
    for (double r : rads)
    {
        const double ratio = r / r_atom;
        int place = 0;
        for (double a : alpha)
            if (ratio > a)
                ++place;
        out.push_back(SG1_LEB_SIZES[static_cast<std::size_t>(place)]);
    }
    return out;
}

std::vector<int> cg1::nwchem(int nuc,
                             const std::vector<double> &rads,
                             int n_ang,
                             const double *radii,
                             const std::vector<int> &lebedev_ngrid_full)
{
    const std::size_t nr = rads.size();
    if (n_ang < 50)
        return std::vector<int>(nr, n_ang);

    // 从 50 开始的切片（等价于 Python: LEBEDEV_NGRID[4:]）
    if (lebedev_ngrid_full.size() < 5)
        throw std::invalid_argument("lebedev_ngrid_full too small");
    std::vector<int> leb_ngrid(lebedev_ngrid_full.begin() + 4, lebedev_ngrid_full.end());
    std::array<int, 5> leb_l{};
    if (n_ang == 50)
    {
        leb_l = {0, 1, 1, 1, 0};
    }
    else
    {
        // 在切片中找 n_ang 的索引
        int idx = -1;
        for (int i = 0; i < static_cast<int>(leb_ngrid.size()); ++i)
        {
            if (leb_ngrid[i] == n_ang)
            {
                idx = i;
                break;
            }
        }
        if (idx < 0)
        {
            throw std::invalid_argument("n_ang not found in lebedev_ngrid_full slice");
        }
        leb_l = {0, 2, idx - 1, idx, idx - 1};
    }

    const auto &alpha = pick_alpha_row(nuc);
    const double r_atom = chemgrid::constants::bragg_radius_bohr((uint32_t)nuc) + 1e-200;

    std::vector<int> out;
    out.reserve(nr);
    for (double r : rads) // rads里的所有r
    {
        const double ratio = r / r_atom;
        int place = 0;
        for (double a : alpha)
            if (ratio > a)
                ++place;
        const int lvl = leb_l[static_cast<std::size_t>(place)];
        out.push_back(leb_ngrid.at(static_cast<std::size_t>(lvl)));
    }
    return out;
}

std::vector<int> cg1::treutler(int /*nuc*/,
                               const std::vector<double> &rads,
                               int n_ang)
{
    const std::size_t nr = rads.size();
    std::vector<int> out(nr);
    const std::size_t t1 = nr / 3;
    const std::size_t t2 = nr / 2;
    for (std::size_t i = 0; i < nr; ++i)
    {
        if (i < t1)
            out[i] = 14;
        else if (i < t2)
            out[i] = 50;
        else
            out[i] = n_ang;
    }
    return out;
}

namespace cg2 = chemgrid::grid;

void cg2::
    gen_atom_grid(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w,
                  std::vector<int> &atom_id,
                  std::vector<int> &atom_nuc,
                  chemgrid::RadialMethod radi_method,
                  int level,
                  chemgrid::PruningMethod prune_method)
{
    for (int atom : atom_nuc)
    {
        printf("atom = %d\n", atom);                               // 拿到每个要处理的元素
        int n_rad = chemgrid::constants::default_rad(atom, level); // 默认径向壳层数
        int n_ang = chemgrid::constants::default_ang(atom, level); // 默认角向点数
        // 生成径向网格 镜像网格最大也就200 没必要GPU上
        std::vector<double> r, dr;
        // 根据方法选择径向网格生成
        switch (radi_method)
        {
        case chemgrid::RadialMethod::TreutlerAhlrichs:
            chemgrid::radial::treutler_ahlrichs(n_rad, atom, true, chemgrid::constants::TA_XI, chemgrid::constants::TA_XI_MAX_Z, r, dr);
            break;
        case chemgrid::RadialMethod::GaussChebyshev:
            chemgrid::radial::gauss_chebyshev(n_rad, r, dr);
            break;
        case chemgrid::RadialMethod::MuraKnowles:
            chemgrid::radial::mura_knowles(n_rad, atom, r, dr);
            break;
        case chemgrid::RadialMethod::Delley:
            chemgrid::radial::delley(n_rad, r, dr);
            break;
        default:
            throw std::invalid_argument("Invalid radial method");
        }
        // 径向权重生成
        constexpr double four_pi = 4.0 * 3.141592653589793238462643383279502884;
        std::vector<double> rad_weight(n_rad);
        for (int i = 0; i < r.size(); i++)
            rad_weight[i] = four_pi * r[i] * r[i] * dr[i];

        // 径向网格裁剪 也是最多200 不需要GPU
        std::vector<int> angs(n_rad);
        switch (prune_method)
        {
        case chemgrid::PruningMethod::None:
            std::fill(angs.begin(), angs.end(), n_ang); // 不裁剪
            break;
        case chemgrid::PruningMethod::SG1:
            angs = chemgrid::prune::sg1(atom, r, n_ang, chemgrid::constants::SG1_RADII_BOHR);
            break;
        case chemgrid::PruningMethod::NWChem:
            angs = chemgrid::prune::nwchem(atom, r, n_ang, chemgrid::constants::BRAGG_RADII_ANG, chemgrid::LEBEDEV_NGRID);
            break;
        case chemgrid::PruningMethod::Treutler:
            angs = chemgrid::prune::treutler(atom, r, n_ang);
            break;
        default:
            throw std::invalid_argument("Invalid pruning method");
            break;
        }
        // 生成角向网格
        //  =========================
        // 生成角向网格并做外积
        // =========================
        // for(int i = 0;i< angs.size();++i)
        // {
        //     printf("index: %d, angs: %d \n",i,angs[i]);
        // }
        // 1) (angs[i], i) 打包并按 n 升序排序，一次性得到唯一 n 及分桶的径向索引
        std::vector<std::pair<int, std::size_t>> pairs;
        pairs.reserve(angs.size());
        for (std::size_t i = 0; i < angs.size(); ++i)
            pairs.emplace_back(angs[i], i);

        std::sort(pairs.begin(), pairs.end(),
                  [](const auto &a, const auto &b)
                  {
                      if (a.first != b.first)
                          return a.first < b.first;
                      return a.second < b.second;
                  });

        std::vector<int> unique_n;
        std::vector<std::vector<std::size_t>> buckets;
        if (!pairs.empty())
        {
            unique_n.push_back(pairs[0].first);
            buckets.push_back({pairs[0].second});
            for (std::size_t k = 1; k < pairs.size(); ++k)
            {
                int v = pairs[k].first;
                std::size_t i = pairs[k].second;
                if (v == unique_n.back())
                {
                    buckets.back().push_back(i);
                }
                else
                {
                    unique_n.push_back(v);
                    buckets.push_back({i});
                }
            }
        }

        // 2) 预估本元素将新增的总点数并扩充外部缓冲的容量，减少多次扩容
        std::size_t add_cap = 0;
        for (std::size_t g = 0; g < unique_n.size(); ++g)
            add_cap += buckets[g].size() * static_cast<std::size_t>(unique_n[g]);

        x.reserve(x.size() + add_cap);
        y.reserve(y.size() + add_cap);
        z.reserve(z.size() + add_cap);
        w.reserve(w.size() + add_cap);

        // 3) 对每个桶做外积：ri × (xk,yk,zk)，权重 wri × wk
        for (std::size_t g = 0; g < unique_n.size(); ++g)
        {
            const int n = unique_n[g];

            // 角向 SoA 网格（接口按你给的使用）
            const auto &G = make_lebedev_grid(n);
            const std::size_t K = G.w.size();
            if (G.x.size() != K || G.y.size() != K || G.z.size() != K)
                throw std::runtime_error("Lebedev grid inconsistent sizes");

            const auto &idx = buckets[g];
            for (std::size_t ii : idx)
            {
                const double ri = r[ii];
                const double wri = rad_weight[ii];
                for (std::size_t kpt = 0; kpt < K; ++kpt)
                {
                    x.push_back(ri * G.x[kpt]);
                    y.push_back(ri * G.y[kpt]);
                    z.push_back(ri * G.z[kpt]);
                    w.push_back(wri * G.w[kpt]);
                    atom_id.push_back(atom); // 记录该点对应的原子核电荷数
                }
            }
        }
    }
}
/****
 * 分子partionion 生成器
 * @param atm_coords 分子中各原子坐标(处理后)
 *
 */
cg2::PartitionOut cg2::get_partition_from_templates(
    const std::vector<std::array<double, 3>> &atm_coords,
    const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
    const std::vector<int> &charges_Z,
    chemgrid::RadiiAdjustMethod radii_adjust_type,
    const double *atomic_radii_by_Z,
    chemgrid::BeckeScheme becke_scheme,
    bool concat)
{
    const int A = static_cast<int>(atm_coords.size()); // 原子个数
    if (A <= 0)
        return {};
    // 监测一下是否小于
    if (static_cast<int>(charges_Z.size()) != A)
        throw std::invalid_argument("charges_Z size mismatch with atm_coords");

    // 0) 半径调整器
    // 半径调整类
    partition::RadiiAdjust adjust;
    if (radii_adjust_type != chemgrid::RadiiAdjustMethod::None) // 不是None的话创建一个调整类
    {
        adjust = chemgrid::partition::RadiiAdjust(charges_Z, atomic_radii_by_Z, chemgrid::constants::BRAGG_MAX_Z, radii_adjust_type);
    }

    // 1) 原子-原子距离矩阵
    const auto atm_dist = compute_atm_dist(atm_coords); // 计算原子间距离
    constexpr double eps = 1e-200;

    // 2) 输出（concat=true：拼接；false：你也可以改成返回 per-atom 结构）
    std::vector<std::array<double, 3>> coords_all; // 所有的权重
    std::vector<double> weights_all;

    // 粗略容量估计：用每个原子模板的点数（取该 Z 对应模板大小）
    std::size_t total_pts = 0;     // 总的点数
    for (int ia = 0; ia < A; ++ia) // 遍历所有原子
    {
        int Z = charges_Z[ia];              // 拿到这个原子对应的电荷数
        auto it = grouped_by_Z.find(Z);     // 获取这个原子的网格
        if (it != grouped_by_Z.end())       // 如果不是空
            total_pts += it->second.size(); // 将这个原子的网格追加进去
    }
    coords_all.reserve(total_pts); // 空间初始化
    weights_all.reserve(total_pts);

    // 3) 逐原子：对每个 ia，取该元素的模板点，平移，然后做 Becke 分区
    for (int ia = 0; ia < A; ++ia)
    {
        const int Z = charges_Z[ia];    // 获取这个原子的电荷数
        auto it = grouped_by_Z.find(Z); // 模板中找到这个原子的网格
        if (it == grouped_by_Z.end())
        {
            // 当前元素没有模板网格，跳过
            continue;
        }

        const auto &tpl = it->second;     // 模板点（相对于原子中心，未平移）
        const std::size_t G = tpl.size(); // 获取原子的大小
        if (G == 0)
            continue;

        // 3.1 平移到分子坐标系；同时取出 tpl 的原始体积权重
        std::vector<std::array<double, 3>> coords(G); // 复制一个
        std::vector<double> vol(G);                   // 权重
        for (std::size_t k = 0; k < G; ++k)
        {
            coords[k] = {tpl[k][0] + atm_coords[ia][0], // 元素平移
                         tpl[k][1] + atm_coords[ia][1],
                         tpl[k][2] + atm_coords[ia][2]};
            vol[k] = tpl[k][3]; // 权重获取
        }

        // 3.2 grid_dist: A×G，点到每个原子的距离
        std::vector<std::vector<double>> grid_dist(A, std::vector<double>(G, 0.0)); // A个原子，每个原子维护一个到G个点的距离并初始化为1
        for (int i = 0; i < A; ++i)
        {
            // 每个原子
            for (std::size_t k = 0; k < G; ++k)
            {
                // 每个点
                // 计算这个点到所有原子的距离
                double dx = coords[k][0] - atm_coords[i][0];
                double dy = coords[k][1] - atm_coords[i][1];
                double dz = coords[k][2] - atm_coords[i][2];
                grid_dist[i][k] = std::sqrt(dx * dx + dy * dy + dz * dz); // 距离计算并维护
            }
        }

        // 3.3 pbecke: A×G，全 1
        std::vector<std::vector<double>> pbecke(A, std::vector<double>(G, 1.0)); // A个原子，每个原子到每个点都有一个becke权重影响值

        // 3.4 上三角原子对遍历，更新 pbecke
        for (int i = 0; i < A; ++i)
        {
            // 遍历
            for (int j = 0; j < i; ++j)
            {
                // 获得这两个原子的距离
                const double Dij = atm_dist[i][j] + eps; // 防止为0
                for (std::size_t k = 0; k < G; ++k)      // 遍历所有点
                {
                    double g = (grid_dist[i][k] - grid_dist[j][k]) / Dij; // 看看这个点到哪个点更近
                    if (!adjust.empty())
                        g = adjust(i, j, g); // 有需要调整就调整
                    g = becke_smooth(g, becke_scheme);
                    pbecke[i][k] *= 0.5 * (1.0 - g);
                    pbecke[j][k] *= 0.5 * (1.0 + g);
                }
            }
        }

        // 3.5 列归一化，并取 ia 行作为该原子贡献，拼接输出
        std::vector<double> colsum(G, 0.0);
        for (std::size_t k = 0; k < G; ++k)
        {
            double s = 0.0;
            for (int i = 0; i < A; ++i)
                s += pbecke[i][k];
            if (s == 0.0)
                s = 1.0; // 防守
            colsum[k] = s;
        }

        for (std::size_t k = 0; k < G; ++k)
        {
            const double wk = vol[k] * pbecke[ia][k] / colsum[k];
            coords_all.push_back(coords[k]);
            weights_all.push_back(wk);
        }
    }

    return {std::move(coords_all), std::move(weights_all)};
}

cg2::PartitionOut cg2::get_partition_from_templates_test1(
    const std::vector<std::array<double, 3>> &atm_coords,
    const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
    const std::vector<int> &charges_Z,
    chemgrid::RadiiAdjustMethod radii_adjust_type,
    const double *atomic_radii_by_Z,
    chemgrid::BeckeScheme becke_scheme,
    bool concat)
{
    const int A = static_cast<int>(atm_coords.size()); // 原子个数
    if (A <= 0)
        return {};
    // 监测一下是否小于
    if (static_cast<int>(charges_Z.size()) != A)
        throw std::invalid_argument("charges_Z size mismatch with atm_coords");

    // 0) 半径调整器
    // 半径调整类
    // partition::RadiiAdjust adjust;
    // if (radii_adjust_type != chemgrid::RadiiAdjustMethod::None) // 不是None的话创建一个调整类
    // {
    //     adjust = chemgrid::partition::RadiiAdjust(charges_Z, atomic_radii_by_Z, chemgrid::constants::BRAGG_MAX_Z, radii_adjust_type);
    // }

    // 对于半径调整器的改进 不使用类直接事先计算
    // 初始化半径向量 //可以水一次
    // 不要GPU做IF 前辈的教导
    std::vector<double> rad(A);
    // cpu进行不用担心if导致的cuda访存
    constexpr double eps = 1e-200;
    switch (radii_adjust_type)
    {
    case chemgrid::RadiiAdjustMethod::Treutler:
    {

        for (int i = 0; i < A; ++i)
        {
            double base = atomic_radii_by_Z[charges_Z[i]]; // 拿到半径
            rad[i] = sqrt(base) + eps;
        }
    }
    break;
    case chemgrid::RadiiAdjustMethod::Becke:
    {
        for (int i = 0; i < A; ++i)
        {
            double base = atomic_radii_by_Z[charges_Z[i]]; // 拿到半径
            rad[i] = base + eps;
        }
    }
    break;
    default:
    {
        // none 的情况
        for (int i = 0; i < A; ++i)
        {
            double base = atomic_radii_by_Z[charges_Z[i]]; // 拿到半径
            rad[i] = base + eps;
        }
    }
    break;
    }
    std::vector<int> a_(A * A); // 原子间互相影响
    for (int i = 0; i < A; ++i)
    {
        double ri = rad[i];
        for (int j = 0; j < A; ++j)
        {
            double rj = rad[j];
            double a_ij = 0.25 * ((rj / ri) - (ri / rj));
            a_[i * A + j] = std::clamp(a_ij, -0.5, 0.5);
        }
    }

    // 1) 原子-原子距离矩阵
    const auto atm_dist = compute_atm_dist(atm_coords); // 计算原子间距离

    // 2) 输出（concat=true：拼接；false：你也可以改成返回 per-atom 结构）
    std::vector<std::array<double, 3>> coords_all; // 所有的权重
    std::vector<double> weights_all;

    // 粗略容量估计：用每个原子模板的点数（取该 Z 对应模板大小）
    std::size_t total_pts = 0;     // 总的点数
    for (int ia = 0; ia < A; ++ia) // 遍历所有原子
    {
        int Z = charges_Z[ia];              // 拿到这个原子对应的电荷数
        auto it = grouped_by_Z.find(Z);     // 获取这个原子的网格
        if (it != grouped_by_Z.end())       // 如果不是空
            total_pts += it->second.size(); // 将这个原子的网格追加进去
    }
    coords_all.reserve(total_pts); // 空间初始化
    weights_all.reserve(total_pts);

    // 3) 逐原子：对每个 ia，取该元素的模板点，平移，然后做 Becke 分区
    for (int ia = 0; ia < A; ++ia)
    {
        const int Z = charges_Z[ia];    // 获取这个原子的电荷数
        auto it = grouped_by_Z.find(Z); // 模板中找到这个原子的网格
        if (it == grouped_by_Z.end())
        {
            // 当前元素没有模板网格，跳过
            continue;
        }

        const auto &tpl = it->second;     // 模板点（相对于原子中心，未平移）
        const std::size_t G = tpl.size(); // 获取原子的大小
        if (G == 0)
            continue;

        // 3.1 平移到分子坐标系；同时取出 tpl 的原始体积权重
        std::vector<std::array<double, 3>> coords(G); // 复制一个
        std::vector<double> vol(G);                   // 权重
        for (std::size_t k = 0; k < G; ++k)
        {
            coords[k] = {tpl[k][0] + atm_coords[ia][0], // 元素平移
                         tpl[k][1] + atm_coords[ia][1],
                         tpl[k][2] + atm_coords[ia][2]};
            vol[k] = tpl[k][3]; // 权重获取
        }

        // 3.2 grid_dist: A×G，点到每个原子的距离
        std::vector<std::vector<double>> grid_dist(A, std::vector<double>(G, 0.0)); // A个原子，每个原子维护一个到G个点的距离并初始化为1
        for (int i = 0; i < A; ++i)
        {
            // 每个原子
            for (std::size_t k = 0; k < G; ++k)
            {
                // 每个点
                // 计算这个点到所有原子的距离
                double dx = coords[k][0] - atm_coords[i][0];
                double dy = coords[k][1] - atm_coords[i][1];
                double dz = coords[k][2] - atm_coords[i][2];
                grid_dist[i][k] = std::sqrt(dx * dx + dy * dy + dz * dz); // 距离计算并维护
            }
        }

        // 3.3 pbecke: A×G，全 1
        std::vector<std::vector<double>> pbecke(A, std::vector<double>(G, 1.0)); // A个原子，每个原子到每个点都有一个becke权重影响值
        
        // 3.4 上三角原子对遍历，更新 pbecke
        for (int i = 0; i < A; ++i)
        {
            // 遍历
            for (int j = 0; j < i; ++j)
            {
                // 获得这两个原子的距离
                const double Dij = atm_dist[i][j] + eps; // 防止为0
                for (std::size_t k = 0; k < G; ++k)      // 遍历所有点
                {
                    double g = (grid_dist[i][k] - grid_dist[j][k]) / Dij; // 看看这个点到哪个点更近
                    // if (!adjust.empty())
                    //     g = adjust(i, j, g); // 有需要调整就调整
                    g = g + (g * g - 1.0) * (-a_[i * A + j]); // 调整
                    // g = becke_smooth(g, becke_scheme);        // 避免cpuif
                    // 闭式多项式与for becke_smooth
                    g = std::clamp(g, -1.0, 1.0);
                    g = 1.5 *g -  0.5*g*g*g;
                    g = 1.5 *g -  0.5*g*g*g;
                    g = 1.5 *g -  0.5*g*g*g;
                    pbecke[i][k] *= 0.5 * (1.0 - g);
                    pbecke[j][k] *= 0.5 * (1.0 + g);
                }
            }
        }

        // 3.5 列归一化，并取 ia 行作为该原子贡献，拼接输出
        std::vector<double> colsum(G, 0.0);
        for (std::size_t k = 0; k < G; ++k)
        {
            double s = 0.0;
            for (int i = 0; i < A; ++i)
                s += pbecke[i][k];
            if (s == 0.0)
                s = 1.0; // 防守
            colsum[k] = s;
        }

        for (std::size_t k = 0; k < G; ++k)
        {
            const double wk = vol[k] * pbecke[ia][k] / colsum[k];
            coords_all.push_back(coords[k]);
            weights_all.push_back(wk);
        }
    }

    return {std::move(coords_all), std::move(weights_all)};
}



// grouped_by_Z: Z -> {{x,y,z,w}...}（每种元素的模板）
// charges_Z:    长度为 natm，charges_Z[ia] = 该原子核电荷 Z
void cg2::build_atm_idx_and_weights_by_Z(
    const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
    const std::vector<int> &charges_Z,
    const std::vector<std::array<double, 3>> &coords_all,
    const std::vector<double> &weights_all,
    std::vector<int> &atm_idx,               // 输出
    std::vector<double> &quadrature_weights) // 输出
{
    const int natm = static_cast<int>(charges_Z.size());
    const std::size_t N = coords_all.size();
    if (weights_all.size() != N)
        throw std::invalid_argument("weights_all size mismatch with coords_all");

    // 预估总点数 = sum_ia |template(Z_ia)|
    std::size_t expected = 0;
    for (int ia = 0; ia < natm; ++ia)
    {
        int Z = charges_Z[ia];
        auto it = grouped_by_Z.find(Z);
        if (it != grouped_by_Z.end())
            expected += it->second.size();
    }
    if (expected != N)
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "Point count mismatch: expected=%zu (sum per-atom templates), N=%zu (coords_all).",
                      expected, N);
        throw std::runtime_error(buf);
    }

    atm_idx.resize(N);
    quadrature_weights = weights_all;

    // 按 ia 顺序，逐段写入 ia 标签（段长 = 该元素模板点数）
    std::size_t p = 0;
    for (int ia = 0; ia < natm; ++ia)
    {
        int Z = charges_Z[ia];
        auto it = grouped_by_Z.find(Z);
        const std::size_t G = (it == grouped_by_Z.end() ? 0 : it->second.size());
        for (std::size_t k = 0; k < G; ++k)
        {
            atm_idx[p + k] = ia;
        }
        p += G;
    }
}