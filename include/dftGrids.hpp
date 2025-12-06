#pragma once
/**
 * @file dftGrids
 * @brief levdev and atom grids for DFT
 */
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <map>
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
#include "basis_set.hpp"
#include "types.hpp"
// grids constant
namespace gansu::dft::chemgrid::constants
{
    // ====== Scalar constants (synonyms kept for readability) =====================

    inline constexpr double GROUP_BOX_SIZE = 1.2;
    inline constexpr double GROUP_BOUNDARY_PENALTY = 4.2;
    inline constexpr int ALIGNMENT_UNIT = 8;
    inline constexpr double CUTOFF = 1e-12;
    inline constexpr double NELEC_ERROR_TOL = 0.02;
    inline constexpr bool ATOM_SPECIFIC_TREUTLER_GRIDS = true;

    inline constexpr double unknown_ang = 1.999999;       // placeholder (Å)
    inline constexpr double BOHR_ANG = 0.52917721092;     // 1 bohr = 0.529177... Å
    inline constexpr double ANG2BOHR = 1.889726124565062; // Å → bohr
    inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    // ====== Tabulated data (stored in Å for readability; accessors return bohr) ==
    //
    // Notes: BRAGG_RADII_ANG[Z] is in Å (index 0 is Ghost/unknown).
    //        The accessor bragg_radius_bohr(Z) multiplies by ANG2BOHR.
    // -----------------------------------------------------------------------------

    inline constexpr double BRAGG_RADII_ANG[] = {
        /*0*/ unknown_ang, // Ghost atom / unknown
        /*1*/ 0.35,
        /*2*/ 1.40, // 1s
        /*3*/ 1.45,
        1.05,
        0.85,
        0.70,
        0.65,
        0.60,
        0.50,
        1.50, // 2s2p
        /*11*/ 1.80,
        1.50,
        1.25,
        1.10,
        1.00,
        1.00,
        1.00,
        1.80, // 3s3p
        /*19*/ 2.20,
        1.80, // 4s
        /*21*/ 1.60,
        1.40,
        1.35,
        1.40,
        1.40,
        1.40,
        1.35,
        1.35,
        1.35,
        1.35, // 3d
        /*31*/ 1.30,
        1.25,
        1.15,
        1.15,
        1.15,
        1.90, // 4p
        /*37*/ 2.35,
        2.00, // 5s
        /*39*/ 1.80,
        1.55,
        1.45,
        1.45,
        1.35,
        1.30,
        1.35,
        1.40,
        1.60,
        1.55, // 4d
        /*49*/ 1.55,
        1.45,
        1.45,
        1.40,
        1.40,
        2.10, // 5p
        /*55*/ 2.60,
        2.15, // 6s
        /*57*/ 1.95,
        1.85,
        1.85,
        1.85,
        1.85,
        1.85,
        1.85, // La, Ce–Eu
        /*64*/ 1.80,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75, // Gd, Tb–Lu
        /*72*/ 1.55,
        1.45,
        1.35,
        1.35,
        1.30,
        1.35,
        1.35,
        1.35,
        1.50, // 5d
        /*81*/ 1.90,
        1.80,
        1.60,
        1.90,
        1.45,
        2.10, // 6p
        /*87*/ 1.80,
        2.15, // 7s
        /*89*/ 1.95,
        1.80,
        1.80,
        1.75,
        1.75,
        1.75,
        1.75,
        /*96*/ 1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        /*103*/ 1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        /*113*/ 1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        1.75,
        /*119? pad*/ // Matches the original Python list; trailing entries use 1.75 Å placeholders
    };
    // “Maximum usable Z” (length − 1; index 0 is Ghost)
    inline constexpr std::size_t BRAGG_MAX_Z =
        sizeof(BRAGG_RADII_ANG) / sizeof(double) - 1;

    // SG1 radii (units = bohr; defined up to Ar only; for Z > 18, fall back to BRAGG)
    inline constexpr double SG1_RADII_BOHR[] = {
        /*0*/ 0.0, /*1*/ 1.0000, /*2*/ 0.5882,
        /*3*/ 3.0769, 2.0513, 1.5385, 1.2308, 1.0256, 0.8791, 0.7692, 0.6838,
        /*11*/ 4.0909, 3.1579, 2.5714, 2.1687, 1.8750, 1.6514, 1.4754, 1.3333};
    inline constexpr std::size_t SG1_MAX_Z =
        sizeof(SG1_RADII_BOHR) / sizeof(double) - 1;

    // Treutler–Ahlrichs ξ (dimensionless; matches your source list)
    // double* xi_table uses these values to control partitioning
    inline constexpr double TA_XI[] = {
        /*0*/ 1.0,                                               // Ghost
        /*1*/ 0.8, /*2*/ 0.9,                                    // 1s
        /*3*/ 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,            // 2s2p
        /*11*/ 1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,           // 3s3p
        /*19*/ 1.5, 1.4,                                         // 4s
        /*21*/ 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, // 3d
        /*31*/ 1.1, 1.0, 0.9, 0.9, 0.9, 0.9,                     // 4p
        /*37*/ 2.000, 1.700,                                     // 5s
        /*39*/ 1.500, 1.500, 1.350, 1.350, 1.250, 1.200, 1.250, 1.300,
        1.500, 1.500,                                                         // 4d
        /*49*/ 1.300, 1.200, 1.200, 1.150, 1.150, 1.150,                      // 5p
        /*55*/ 2.500, 2.200,                                                  // 6s
        /*57*/ 2.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,               // La, Ce–Eu
        /*64*/ 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,        // Gd–Lu
        /*72*/ 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, // 5d
        /*81*/ 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,                      // 6p
        /*87*/ 2.500, 2.100,                                                  // 7s
        /*89*/ 3.685, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500,
        1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500, 1.500};
    inline constexpr std::size_t TA_XI_MAX_Z =
        sizeof(TA_XI) / sizeof(double) - 1;

    // ====== Safe accessors (always return bohr; include sensible fallbacks) =======
    //
    // Design notes:
    // 1) BRAGG: Å → bohr conversion happens inside the function; index 0 (Ghost) uses unknown_ang.
    // 2) SG1: defined only up to Ar; out-of-range falls back to BRAGG.
    // 3) TA ξ: if missing, falls back to 1.5 (empirical default).
    // -----------------------------------------------------------------------------

    double bragg_radius_bohr(std::uint32_t Z);

    double sg1_radius_bohr(std::uint32_t Z);

    inline double ta_xi(std::uint32_t Z, double fallback = 1.5)
    {
        const std::size_t n = sizeof(TA_XI) / sizeof(double);
        if (Z < n)
        {
            const double v = TA_XI[Z];
            return (v == v) ? v : fallback; // NaN check without <cmath> isnan
        }
        return fallback;
    }

    // ====== Default grid sizes (radial/angular) and Lebedev order mapping =========
    inline constexpr int RAD_GRIDS[10][7] = {
        /*period→         0    1    2    3    4    5    6  */
        /*level=0*/ {10, 15, 20, 30, 35, 40, 50},
        {30, 40, 50, 60, 65, 70, 75},
        {40, 60, 65, 75, 80, 85, 90},
        {50, 75, 80, 90, 95, 100, 105},
        {60, 90, 95, 105, 110, 115, 120},
        {70, 105, 110, 120, 125, 130, 135},
        {80, 120, 125, 135, 140, 145, 150},
        {90, 135, 140, 150, 155, 160, 165},
        {100, 150, 155, 165, 170, 175, 180},
        /*level=9*/ {200, 200, 200, 200, 200, 200, 200}};

    inline constexpr int ANG_ORDER[10][7] = {
        /*period→         0    1    2    3    4    5    6  */
        /*level=0*/ {11, 15, 17, 17, 17, 17, 17},
        {17, 23, 23, 23, 23, 23, 23},
        {23, 29, 29, 29, 29, 29, 29},
        {29, 29, 35, 35, 35, 35, 35},
        {35, 41, 41, 41, 41, 41, 41},
        {41, 47, 47, 47, 47, 47, 47},
        {47, 53, 53, 53, 53, 53, 53},
        {53, 59, 59, 59, 59, 59, 59},
        {59, 59, 59, 59, 59, 59, 59},
        /*level=9*/ {65, 65, 65, 65, 65, 65, 65}};

    // Period segmentation thresholds (Z): (0..2], (3..10], (11..18], (19..36], (37..54], (55..86], (87..118]
    inline constexpr int PERIOD_THRESHOLDS[7] = {2, 10, 18, 36, 54, 86, 118};

    // ---- Helper: Z → period index (0..6); clamp out-of-range to 6 ----
    inline constexpr int period_index(unsigned Z)
    {
        int idx = 0;
        for (int i = 0; i < 7; ++i)
        {
            if (static_cast<int>(Z) > PERIOD_THRESHOLDS[i])
                ++idx;
        }
        return (idx < 0 ? 0 : (idx > 6 ? 6 : idx));
    }

    // ---- Helper: clamp level to 0..9 ----
    inline constexpr int clamp_level(int level)
    {
        return (level < 0 ? 0 : (level > 9 ? 9 : level));
    }

    // ====== Public accessors ======================================================

    // Default number of radial shells (level: 0..9; default = 3)
    inline int default_rad(unsigned Z, int level = 3)
    {
        const int L = clamp_level(level);
        const int P = period_index(Z);
        return RAD_GRIDS[L][P];
    }

    // Default number of angular points (Lebedev order; level: 0..9; default = 3)
    inline int default_ang(unsigned Z, int level = 3)
    {
        const int L = clamp_level(level);
        const int P = period_index(Z);
        std::map<int, int> LEBEDEV_ORDER = {
            {0, 1},
            {3, 6},
            {5, 14},
            {7, 26},
            {9, 38},
            {11, 50},
            {13, 74},
            {15, 86},
            {17, 110},
            {19, 146},
            {21, 170},
            {23, 194},
            {25, 230},
            {27, 266},
            {29, 302},
            {31, 350},
            {35, 434},
            {41, 590},
            {47, 770},
            {53, 974},
            {59, 1202},
            {65, 1454},
            {71, 1730},
            {77, 2030},
            {83, 2354},
            {89, 2702},
            {95, 3074},
            {101, 3470},
            {107, 3890},
            {113, 4334},
            {119, 4802},
            {125, 5294},
            {131, 5810}};

        return LEBEDEV_ORDER[ANG_ORDER[L][P]];
    }

}

// lebdev //
namespace gansu::dft::chemgrid
{
    enum class RadiiAdjustMethod : int
    {
        None = 0,     ///< No radius adjustment
        Treutler = 1, ///< Treutler atomic radius adjustment
        Becke = 2     ///< Becke atomic radius adjustment
    };

    // ---- Radial Grid Methods ----
    enum class RadialMethod : int
    {
        Delley = 1,          ///< Delley radial grid (log2 algorithm)
        MuraKnowles = 2,     ///< Mura-Knowles radial grid (log3 algorithm)
        GaussChebyshev = 3,  ///< Gauss-Chebyshev radial grid
        TreutlerAhlrichs = 4 ///< Treutler-Ahlrichs radial grid (M4)
    };

    // ---- Becke Scheme Methods ----
    enum class BeckeScheme : int
    {
        Original = 1, ///< Original Becke scheme
        Stratmann = 2 ///< Stratmann improved scheme
    };

    // ---- Pruning Methods ----
    enum class PruningMethod : int
    {
        None = 0,    ///< No pruning
        SG1 = 1,     ///< SG1 pruning scheme
        NWChem = 2,  ///< NWChem pruning scheme
        Treutler = 3 ///< Treutler pruning scheme
    };

    struct AngGrid
    {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        std::vector<double> w;

        std::size_t size() const noexcept { return w.size(); }
        void reserve(std::size_t n)
        {
            x.reserve(n);
            y.reserve(n);
            z.reserve(n);
            w.reserve(n);
        }
        void push(double xi, double yi, double zi, double wi)
        {
            x.push_back(xi);
            y.push_back(yi);
            z.push_back(zi);
            w.push_back(wi);
        }
    };
    // Lebedev 点数支持：返回可用点数列表（用于参数检查/修剪选择）
    const std::vector<int> &lebedev_supported_sizes();
    const std::vector<int> LEBEDEV_NGRID = {
        6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350,
        434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074,
        3470, 3890, 4334, 4802, 5294, 5810};
    // 生成指定点数的 Lebedev 角向网格（单位球面，权重与表保持一致）
    AngGrid make_lebedev_grid(int n_points);

}

namespace gansu::dft::chemgrid
{
    // radial

    // Delley (log2)
    void delley(int n, std::vector<double> &r, std::vector<double> &dr);

    // Mura–Knowles (log3)
    void mura_knowles(int n, int charge, std::vector<double> &r, std::vector<double> &dr);

    // Gauss–Chebyshev
    void gauss_chebyshev(int n, std::vector<double> &r, std::vector<double> &dr);

    // Treutler–Ahlrichs (M4)
    // - xi_table 可为空：若为空则统一使用 xi=1.0
    // - atom_specific=true 时若表存在则用表中 xi[Z]；否则 xi=1.0
    // 来源于 constant 中的 TA_XI 和 TA_XI_MAX_Z
    void treutler_ahlrichs(int n, int Z,
                           bool atom_specific,
                           const double *xi_table, std::size_t xi_size,
                           std::vector<double> &r, std::vector<double> &dr);

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
}

namespace gansu::dft::chemgrid
{
    void gen_atom_grid(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w,
                       std::vector<int> &atom_id,
                       std::vector<int> &atom_nuc,
                       chemgrid::RadialMethod radi_method = chemgrid::RadialMethod::TreutlerAhlrichs,
                       int level = 3,
                       chemgrid::PruningMethod prune_method = chemgrid::PruningMethod::NWChem);
    using PartitionOut = std::pair<std::vector<std::array<double, 3>>,
                                   std::vector<double>>; // (xyz, w)

    PartitionOut get_partition(
        const std::vector<std::array<double, 3>> &atm_coords,                            // A×3
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z, // templates
        const std::vector<int> &charges_Z,                                               // A
        const double *atomic_radii_by_Z                                                  // [Zmax+1] or nullptr
    );
    void build_atm_idx_and_weights_by_Z(
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        const std::vector<std::array<double, 3>> &coords_all,
        const std::vector<double> &weights_all,
        std::vector<int> &atm_idx, // output
        std::vector<double> &quadrature_weights);
    using Coord3 = std::array<double, 3>;

    // Compute the bounding box and the actual number of boxes (mirrors the Python logic)
    struct BoxGrid
    {
        std::array<double, 3> bmin;
        std::array<double, 3> bmax;
        std::array<int, 3> nbox;        // number of boxes along each axis (>= 1)
        std::array<double, 3> box_size; // actual box size along each axis
    };

    // Derive binning parameters from atom coordinates + box_size
    inline BoxGrid make_boxgrid(const std::vector<Coord3> &atom_coords,
                                double group_boundary_penalty,
                                double req_box_size)
    {
        if (atom_coords.empty())
            throw std::invalid_argument("make_boxgrid: atom_coords empty");

        // Min/max over atom coordinates
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

        // Pad the bounds
        std::array<double, 3> bmin{amin[0] - group_boundary_penalty,
                                   amin[1] - group_boundary_penalty,
                                   amin[2] - group_boundary_penalty};
        std::array<double, 3> bmax{amax[0] + group_boundary_penalty,
                                   amax[1] + group_boundary_penalty,
                                   amax[2] + group_boundary_penalty};

        // Boxes per axis: round((bmax - bmin) / req_box_size), at least 1
        std::array<int, 3> nbox;
        std::array<double, 3> L, h;
        for (int d = 0; d < 3; ++d)
        {
            L[d] = bmax[d] - bmin[d];
            const double nb = std::max(1.0, std::round(L[d] / req_box_size));
            nbox[d] = static_cast<int>(nb);
            h[d] = L[d] / static_cast<double>(nbox[d]); // actual box size
        }

        return BoxGrid{bmin, bmax, nbox, h};
    }

    // Build indices grouped by (ix, iy, iz) in lexicographic order (stable; matches Python)
    // Allowed index range is [-1, nbox[d]]; we map it to [0, nbox[d] + 1] by adding +1
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

        // Construct the box grid
        const BoxGrid bg = make_boxgrid(atom_coords, boundary_pad, req_box_size);
        const int nx = bg.nbox[0], ny = bg.nbox[1], nz = bg.nbox[2];
        const int sx = nx + 2, sy = ny + 2, sz = nz + 2; // +2 to accommodate the mapping of [-1, nbox]

        // Total number of 3D buckets: (nx + 2) * (ny + 2) * (nz + 2)
        const std::size_t nbuckets = static_cast<std::size_t>(sx) * sy * sz;
        std::vector<std::vector<std::size_t>> buckets(nbuckets);

        // Insert points into buckets (stable: preserve input order)
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

            // In Python: values < -1 are set to -1; values > nbox are set to nbox
            int ix_c = clamp(static_cast<int>(ix), -1, nx);
            int iy_c = clamp(static_cast<int>(iy), -1, ny);
            int iz_c = clamp(static_cast<int>(iz), -1, nz);

            // Map to [0..nx+1], etc.
            const int I = ix_c + 1;
            const int J = iy_c + 1;
            const int K = iz_c + 1;

            const std::size_t bid =
                static_cast<std::size_t>(I) +
                static_cast<std::size_t>(sx) * (static_cast<std::size_t>(J) +
                                                static_cast<std::size_t>(sy) * static_cast<std::size_t>(K));
            buckets[bid].push_back(i);
        }

        // Traverse buckets in (ix, iy, iz) lexicographic order and concatenate indices
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
                    idx.insert(idx.end(), b.begin(), b.end()); // stable concatenation
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
            coords.push_back(Coord3{1e-4, 1e-4, 1e-4}); // dummy point coordinate
            weights.push_back(0.0);                     // final weight = 0
            quad_w.push_back(0.0);                      // volumetric weight = 0
            atm_idx.push_back(-1);                      // no owner (easy to skip)
        }
        return pad;
    }

    void evaluate_aos_on_grids_gpu_raw(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao_values, // ← Output: pointer provided by the caller
        int ngrids,
        int nao);
    void evaluate_aos_gpu_shell_grouped(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao_values, // [nao x ngrids]
        int ngrids,
        int nao);
}
// output API
struct AngularMomentumKey
{
    int lx, ly, lz;

    bool operator<(const AngularMomentumKey &other) const
    {
        if (lx != other.lx)
            return lx < other.lx;
        if (ly != other.ly)
            return ly < other.ly;
        return lz < other.lz;
    }

    bool operator==(const AngularMomentumKey &other) const
    {
        return lx == other.lx && ly == other.ly && lz == other.lz;
    }
};

// ============================================================
// Grouped Ao data for GPU evaluation
// ============================================================
struct ShellData
{
    int atom_idx;
    int l;
    double fac;                      // sqrt((2l+1)/(4π))
    int nprim;
    std::vector<double> exps;
    std::vector<double> coeffs;
    std::vector<int> ao_indices;     // Which AOs belong to this shell
    std::vector<int> lx_list;        // lx for each AO
    std::vector<int> ly_list;        // ly for each AO
    std::vector<int> lz_list;        // lz for each AO
};


namespace gansu::dft
{

    std::pair<std::vector<std::array<double, 3>>, std::vector<double>> dft_gen_grid(std::vector<int> charges, std::vector<std::array<double, 3>> atm_coords);
    std::map<int, std::vector<atom_AO>> get_normalized_atom_basis(
        const PrimitiveShell *shells_ptr,
        int bsisnum,
        const Atom *h_atoms, //
        int nAtom);
    AOGrids dft_gen_ao(std::map<int, std::vector<atom_AO>> basis, std::vector<int> charges, std::vector<std::array<double, 3>> &atm_coords, std::vector<std::array<double, 3>> coords);

}