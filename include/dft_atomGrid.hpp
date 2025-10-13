#pragma once

// 径向网格生成
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <array>
#include <cmath>
#include <memory>
#include <cstring>
#include <numeric>
#include <thread>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include "dft_constant.hpp"
#include "dft_lebedev.hpp"

namespace chemgrid::radial
{

    // Delley (log2)
    void delley(int n, std::vector<double> &r, std::vector<double> &dr);

    // Mura–Knowles (log3)
    void mura_knowles(int n, int charge, std::vector<double> &r, std::vector<double> &dr);

    // Gauss–Chebyshev
    void gauss_chebyshev(int n, std::vector<double> &r, std::vector<double> &dr);

    // Treutler–Ahlrichs (M4)
    // - xi_table 可为空：若为空则统一使用 xi=1.0
    // - atom_specific=true 时若表存在则用表中 xi[Z]；否则 xi=1.0
    // 来源于 dft_constant.hpp 中的 TA_XI 和 TA_XI_MAX_Z
    void treutler_ahlrichs(int n, int Z,
                           bool atom_specific,
                           const double *xi_table, std::size_t xi_size,
                           std::vector<double> &r, std::vector<double> &dr);

} // namespace chemgrid::radial

namespace chemgrid::prune
{

    // SG1：返回每个半径对应的 Lebedev 点数
    std::vector<int> sg1(int nuc,
                         const std::vector<double> &rads,
                         int n_ang, // 未用，保持一致
                         const double *radii);

    // NWChem：需要完整的 Lebedev 点数表（如 [6,14,26,38,50,74,86,...]）
    std::vector<int> nwchem(int nuc,
                            const std::vector<double> &rads,
                            int n_ang,
                            const double *radii,
                            const std::vector<int> &lebedev_ngrid_full);

    // Treutler：分段 14/50/n_ang
    std::vector<int> treutler(int nuc,
                              const std::vector<double> &rads,
                              int n_ang);

} // namespace chemgrid::prune

namespace chemgrid::grid
{

    void gen_atom_grid(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w,
                       std::vector<int> &atom_id,
                       std::vector<int> &atom_nuc,
                       chemgrid::RadialMethod radi_method = chemgrid::RadialMethod::TreutlerAhlrichs,
                       int level = 3,
                       chemgrid::PruningMethod prune_method = chemgrid::PruningMethod::NWChem);
    using PartitionOut = std::pair<std::vector<std::array<double, 3>>,
                                   std::vector<double>>; // (xyz, w)
    // 单独使用的伴生becke方法
    inline double becke_smooth(double g, BeckeScheme scheme)
    {
        g = std::clamp(g, -1.0, 1.0);
        auto p3 = [](double x)
        { return 1.5 * x - 0.5 * x * x * x; };
        switch (scheme)
        {
        case BeckeScheme::Original:
            g = p3(g);
            g = p3(g);
            g = p3(g);
            return g;
        case BeckeScheme::Stratmann:
            g = p3(g);
            g = p3(g);
            return g;
        default:
            return g;
        }
    }
    // 伴生的原子间距
    inline std::vector<std::vector<double>>
    compute_atm_dist(const std::vector<std::array<double, 3>> &atm_coords)
    {
        const int A = static_cast<int>(atm_coords.size());
        std::vector<std::vector<double>> D(A, std::vector<double>(A, 0.0));
        for (int i = 0; i < A; ++i)
        {
            for (int j = 0; j < A; ++j)
            {
                double dx = atm_coords[i][0] - atm_coords[j][0];
                double dy = atm_coords[i][1] - atm_coords[j][1];
                double dz = atm_coords[i][2] - atm_coords[j][2];
                D[i][j] = std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        return D;
    }
    PartitionOut get_partition_from_templates(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool concat = true);
    PartitionOut get_partition_from_templates_test1(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool concat = true);
    // GPU版本
    PartitionOut get_partition_from_templates_GPU(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool concat = true);
    // GPU+临近剪枝方法
    PartitionOut get_partition_from_templates_GPU_nbr(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool concat = true);
    // GPU+临近剪枝方法2
    PartitionOut get_partition_from_templates_GPU_nbr2(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_nbr2_1kernel_adaptive(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    // GPU+临近剪枝方法2+流式处理
    PartitionOut get_partition_from_templates_GPU_nbr2_stream(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    // GPU+临近剪枝方法2+多GPU
    PartitionOut get_partition_from_templates_GPU_hbr2_mulGpu(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_nbr2_1kernel_map_bucket(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme /*becke_scheme*/,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_nbr2_1kernel_map(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_nbr2_multistream(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/,
        int num_streams = 4); // 新增参数：stream数量
    // GPU+临近剪枝方法+扁平化内存
    // 919 增加 单kernel的方法

    PartitionOut get_partition_from_templates_GPU_nbr2_1kernel_atom(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_nbr2_1kernel_atom_MUL(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_nbr2_1kernel_atom_MUL_O(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme /*becke_scheme*/,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_1k(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_boxL(
        const std::vector<std::array<double, 3>> &atm_coords,                            // A×3
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z, // Z -> [x,y,z,w]（原子模板，已拼接）
        const std::vector<int> &charges_Z,                                               // A
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z, // Z->radius
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/,
        // --- 关键新参数（可放到常量或配置处）---
        double BOX_SIZE,   // 例如 chemgrid::constants::GROUP_BOX_SIZE
        double BOX_MARGIN, // 例如 chemgrid::constants::GROUP_BOUNDARY_PENALTY
        int L_near         // 例如 32/48/64
    );
    PartitionOut get_partition_from_templates_GPU_nbr2_1kernel_fullpairs_tiled(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme /*becke_scheme*/, // 此版固定 3 次平滑
        bool                                    /*concat*/
    );
    // 原子分组归属整理
    void build_atm_idx_and_weights_by_Z(
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        const std::vector<std::array<double, 3>> &coords_all,
        const std::vector<double> &weights_all,
        std::vector<int> &atm_idx,                // 输出
        std::vector<double> &quadrature_weights); // 输出
    // 改进后的主函数：基于真实分布的分组策略
    PartitionOut get_partition_from_templates_GPU_neighbor_grouped(
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        chemgrid::RadiiAdjustMethod radii_adjust_type,
        const double *atomic_radii_by_Z,
        chemgrid::BeckeScheme becke_scheme,
        bool /*concat*/);
    PartitionOut get_partition_from_templates_GPU_warp_ownerOnly(
        const std::vector<std::array<double, 3>> &atm_coords,                            // A×3
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z, // templates
        const std::vector<int> &charges_Z,                                               // A
        const double *atomic_radii_by_Z                                                  // [Zmax+1] or nullptr
    );

}

namespace chemgrid::grid
{
    using Coord3 = std::array<double, 3>;

    // 计算边界盒及实际盒子数（按 Python 逻辑重现）
    struct BoxGrid
    {
        std::array<double, 3> bmin;
        std::array<double, 3> bmax;
        std::array<int, 3> nbox;        // 每轴盒子个数（>=1）
        std::array<double, 3> box_size; // 实际盒子尺寸
    };

    // 根据原子坐标 + box_size 计算分箱参数
    inline BoxGrid make_boxgrid(const std::vector<Coord3> &atom_coords,
                                double group_boundary_penalty,
                                double req_box_size)
    {
        if (atom_coords.empty())
            throw std::invalid_argument("make_boxgrid: atom_coords empty");

        // 原子坐标 min/max
        std::array<double, 3> amin{+std::numeric_limits<double>::infinity(),
                                   +std::numeric_limits<double>::infinity(),
                                   +std::numeric_limits<double>::infinity()};
        std::array<double, 3> amax{-std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()};
        for (const auto &r : atom_coords)
        {
            for (int d = 0; d < 3; ++d)
            {
                amin[d] = std::min(amin[d], r[d]);
                amax[d] = std::max(amax[d], r[d]);
            }
        }

        // 扩边
        std::array<double, 3> bmin{amin[0] - group_boundary_penalty,
                                   amin[1] - group_boundary_penalty,
                                   amin[2] - group_boundary_penalty};
        std::array<double, 3> bmax{amax[0] + group_boundary_penalty,
                                   amax[1] + group_boundary_penalty,
                                   amax[2] + group_boundary_penalty};

        // 每轴盒子个数： round((bmax-bmin)/req_box_size)，至少 1
        std::array<int, 3> nbox;
        std::array<double, 3> L, h;
        for (int d = 0; d < 3; ++d)
        {
            L[d] = bmax[d] - bmin[d];
            const double nb = std::max(1.0, std::round(L[d] / req_box_size));
            nbox[d] = static_cast<int>(nb);
            h[d] = L[d] / static_cast<double>(nbox[d]); // 实际盒子尺寸
        }

        return BoxGrid{bmin, bmax, nbox, h};
    }

    // 生成按 (ix,iy,iz) 字典序分组后的 idx（稳定，等价于 Python 逻辑）
    // 允许索引范围为 [-1, nbox[d]]，通过 +1 偏移映射到 [0, nbox[d]+1]
    inline std::vector<std::size_t> arg_group_grids_cpu(
        const std::vector<Coord3> &coords,      // N×3
        const std::vector<Coord3> &atom_coords, // A×3
        double req_box_size,                    // GROUP_BOX_SIZE (1.2)
        double boundary_pad                     // GROUP_BOUNDARY_PENALTY (4.2)
    )
    {
        const std::size_t N = coords.size();
        std::vector<std::size_t> idx;
        idx.reserve(N);

        // 构造盒子网格
        const BoxGrid bg = make_boxgrid(atom_coords, boundary_pad, req_box_size);
        const int nx = bg.nbox[0], ny = bg.nbox[1], nz = bg.nbox[2];
        const int sx = nx + 2, sy = ny + 2, sz = nz + 2; // +2 以容纳 [-1, nbox] 的映射

        // 三维桶容量： (nx+2)*(ny+2)*(nz+2)
        const std::size_t nbuckets = static_cast<std::size_t>(sx) * sy * sz;
        std::vector<std::vector<std::size_t>> buckets(nbuckets);

        // 投点到桶（稳定：保留原始顺序）
        auto clamp = [](int v, int lo, int hi)
        { return std::max(lo, std::min(hi, v)); };

        for (std::size_t i = 0; i < N; ++i)
        {
            const double fx = (coords[i][0] - bg.bmin[0]) / bg.box_size[0];
            const double fy = (coords[i][1] - bg.bmin[1]) / bg.box_size[1];
            const double fz = (coords[i][2] - bg.bmin[2]) / bg.box_size[2];

            // floor → int
            long ix = static_cast<long>(std::floor(fx));
            long iy = static_cast<long>(std::floor(fy));
            long iz = static_cast<long>(std::floor(fz));

            // Python 里会把 < -1 的设为 -1；> nbox 的设为 nbox
            int ix_c = clamp(static_cast<int>(ix), -1, nx);
            int iy_c = clamp(static_cast<int>(iy), -1, ny);
            int iz_c = clamp(static_cast<int>(iz), -1, nz);

            // 映射到 [0..nx+1] 等
            const int I = ix_c + 1;
            const int J = iy_c + 1;
            const int K = iz_c + 1;

            const std::size_t bid =
                static_cast<std::size_t>(I) +
                static_cast<std::size_t>(sx) * (static_cast<std::size_t>(J) +
                                                static_cast<std::size_t>(sy) * static_cast<std::size_t>(K));
            buckets[bid].push_back(i);
        }

        // 以 (ix,iy,iz) 字典序遍历桶，连接出 idx
        idx.clear();
        idx.reserve(N);
        for (int K = 0; K < sz; ++K)
        {
            for (int J = 0; J < sy; ++J)
            {
                for (int I = 0; I < sx; ++I)
                {
                    const std::size_t bid =
                        static_cast<std::size_t>(I) +
                        static_cast<std::size_t>(sx) * (static_cast<std::size_t>(J) +
                                                        static_cast<std::size_t>(sy) * static_cast<std::size_t>(K));
                    const auto &b = buckets[bid];
                    idx.insert(idx.end(), b.begin(), b.end()); // 稳定连接
                }
            }
        }
        return idx;
    }
    template <class T>
    inline void reorder_by_index(const std::vector<std::size_t> &idx,
                                 const std::vector<T> &in,
                                 std::vector<T> &out)
    {
        const std::size_t N = idx.size();
        if (in.size() != N)
            throw std::invalid_argument("reorder_by_index: size mismatch");
        out.resize(N);
        for (std::size_t i = 0; i < N; ++i)
            out[i] = in[idx[i]];
    }
    template <class T>
    inline void reorder_by_index_inplace(const std::vector<std::size_t> &idx,
                                         std::vector<T> &a)
    {
        std::vector<T> tmp;
        tmp.reserve(a.size());
        reorder_by_index(idx, a, tmp);
        a.swap(tmp);
    }

    inline std::size_t padding_size(std::size_t n, std::size_t alignment)
    {
        if (alignment <= 1)
            return 0;
        std::size_t r = n % alignment;
        return r ? (alignment - r) : 0;
    }

    inline std::size_t pad_grids_cpp(std::vector<Coord3> &coords,
                                     std::vector<double> &weights,
                                     std::vector<int> &atm_idx,
                                     std::vector<double> &quad_w,
                                     std::size_t alignment)
    {
        const std::size_t N = coords.size();
        const std::size_t pad = padding_size(N, alignment);
        if (pad == 0)
            return 0;

        coords.reserve(N + pad);
        weights.reserve(N + pad);
        atm_idx.reserve(N + pad);
        quad_w.reserve(N + pad);

        for (std::size_t k = 0; k < pad; ++k)
        {
            coords.push_back(Coord3{1e-4, 1e-4, 1e-4}); // 虚点坐标
            weights.push_back(0.0);                     // 最终权重为0
            quad_w.push_back(0.0);                      // 体积权重为0
            atm_idx.push_back(-1);                      // 无归属（便于跳过）
        }
        return pad;
    }
}
namespace chemgrid::partition
{

    class RadiiAdjust
    {
    public:
        RadiiAdjust() = default;

        // radii: 以 Z 为索引的一维表（长度 >= max(Z)+1）
        // charges: 分子内每个原子的核电荷 Z
        RadiiAdjust(const std::vector<int> &charges,
                    const double *atomic_radii_by_Z,
                    std::size_t radii_len,
                    chemgrid::RadiiAdjustMethod type)
        {
            init(charges, atomic_radii_by_Z, radii_len, type); // 初始化 原子类型与半径
        }

        RadiiAdjust(const std::vector<int> &charges,
                    const std::vector<double> &atomic_radii_by_Z,
                    chemgrid::RadiiAdjustMethod type)
        {
            if (atomic_radii_by_Z.empty())
            {
                throw std::invalid_argument("atomic_radii_by_Z is empty");
            }
            init(charges, atomic_radii_by_Z.data(), atomic_radii_by_Z.size(), type);
        }

        // 可调用：返回 g' = g + (g^2 - 1) * (-a_ij)
        inline double operator()(int i, int j, double g) const noexcept
        {
            // 执行的核心步骤就是 获得a_
            if (n_ <= 0 || type_ == chemgrid::RadiiAdjustMethod::None)
                return g;
            const double aij = a_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(j)];
            const double g2m1 = g * g - 1.0;
            return g + g2m1 * (-aij);
        }

        inline int size() const noexcept { return n_; }
        inline chemgrid::RadiiAdjustMethod type() const noexcept { return type_; }
        inline bool empty() const noexcept { return n_ <= 0; }

    private:
        static inline double clamp(double v, double lo, double hi) noexcept
        {
            return v < lo ? lo : (v > hi ? hi : v);
        }

        void init(const std::vector<int> &charges,
                  const double *radii_by_Z, std::size_t radii_len,
                  chemgrid::RadiiAdjustMethod type)
        {
            type_ = type;
            n_ = static_cast<int>(charges.size());                     // 原子个数
            if (n_ <= 0 || type_ == chemgrid::RadiiAdjustMethod::None) // 输入有误
            {
                a_.clear();
                return;
            }

            // 1) 构造 rad[i]
            constexpr double eps = 1e-200;
            std::vector<double> rad(static_cast<std::size_t>(n_)); // 半径
            for (int i = 0; i < n_; ++i)                           // 遍历原子
            {
                const int Z = charges[static_cast<std::size_t>(i)];    // 核电荷数
                if (Z < 0 || static_cast<std::size_t>(Z) >= radii_len) // 超出范围
                {
                    throw std::out_of_range("RadiiAdjust: Z out of range for atomic_radii_by_Z");
                }
                const double base = radii_by_Z[static_cast<std::size_t>(Z)]; // 获取半径
                double ri;
                switch (type_) // 跟据类型选择半径调整方法
                {
                case chemgrid::RadiiAdjustMethod::Treutler:
                    ri = std::sqrt(base) + eps; // sqrt + eps
                    break;
                case chemgrid::RadiiAdjustMethod::Becke:
                    ri = base + eps; // identity + eps
                    break;
                default: // None 已在前面返回
                    ri = base + eps;
                    break;
                }
                // 防御性：避免非正值
                if (!(ri > 0.0))
                    ri = eps;
                rad[static_cast<std::size_t>(i)] = ri; // 把处理好的半径放进去
            }
            // 思考，是否可以事先计算这个r_i因为在cpu中不影响速度

            // 2) 构造 a_ij（行主序 n×n）： a_ij = 0.25*( rj/ri - ri/rj )，再截断 [-0.5, 0.5]
            a_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_), 0.0); // A*A的 矩阵，调整后的原子间距
            for (int i = 0; i < n_; ++i)
            {
                const double ri = rad[static_cast<std::size_t>(i)]; // 取得这个原子的半径
                for (int j = 0; j < n_; ++j)
                {
                    const double rj = rad[static_cast<std::size_t>(j)];                                                 // 取得B原子的半径
                    double aij = 0.25 * ((rj / ri) - (ri / rj));                                                        // 处理
                    aij = clamp(aij, -0.5, 0.5);                                                                        // 保证上下限
                    a_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(j)] = aij; // 1维化
                }
            }
        }

    private:
        std::vector<double> a_; // a[i*n + j]
        int n_ = 0;
        chemgrid::RadiiAdjustMethod type_ = chemgrid::RadiiAdjustMethod::None;
    };

} // namespace chemgrid::partition