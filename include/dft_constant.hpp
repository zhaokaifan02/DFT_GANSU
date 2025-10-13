#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <cstdio>
#include "dft_lebedev.hpp"
namespace chemgrid::constants
{

    // ====== 标量常量（同名/同义） =====================================

    inline constexpr double GROUP_BOX_SIZE = 1.2;
    inline constexpr double GROUP_BOUNDARY_PENALTY = 4.2;
    inline constexpr int ALIGNMENT_UNIT = 8;
    inline constexpr double CUTOFF = 1e-12;
    inline constexpr double NELEC_ERROR_TOL = 0.02;
    inline constexpr bool ATOM_SPECIFIC_TREUTLER_GRIDS = true;

    inline constexpr double unknown_ang = 1.999999;    // 占位（Å）
    inline constexpr double BOHR_ANG = 0.52917721092;  // 1 bohr = 0.529177... Å
    inline constexpr double ANG2BOHR = 1.889726124565062; // Å → bohr
    inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    // ====== 表数据（保持“原始单位=Å”的直观性；使用者通过取值函数拿 bohr） ======
    //
    // 说明：BRAGG_RADII_ANG[Z] 以 Å 存储（索引 0 为 Ghost/unknown）。
    //      访问函数 bragg_radius_bohr(Z) 会自动乘 ANG2BOHR。
    // ---------------------------------------------------------------------------

    inline constexpr double BRAGG_RADII_ANG[] = {
        /*0*/ unknown_ang,                                                 // Ghost atom / unknown
        /*1*/ 0.35, /*2*/ 1.40,                                            // 1s
        /*3*/ 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,              // 2s2p
        /*11*/ 1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             // 3s3p
        /*19*/ 2.20, 1.80,                                                 // 4s
        /*21*/ 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, // 3d
        /*31*/ 1.30, 1.25, 1.15, 1.15, 1.15, 1.90,                         // 4p
        /*37*/ 2.35, 2.00,                                                 // 5s
        /*39*/ 1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, // 4d
        /*49*/ 1.55, 1.45, 1.45, 1.40, 1.40, 2.10,                         // 5p
        /*55*/ 2.60, 2.15,                                                 // 6s
        /*57*/ 1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   // La, Ce–Eu
        /*64*/ 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             // Gd, Tb–Lu
        /*72*/ 1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50,       // 5d
        /*81*/ 1.90, 1.80, 1.60, 1.90, 1.45, 2.10,                         // 6p
        /*87*/ 1.80, 2.15,                                                 // 7s
        /*89*/ 1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
        /*96*/ 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        /*103*/ 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        /*113*/ 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        /*119? pad*/ // 以上与原 Python 列表一一对应（末段多为 1.75 Å 占位）
    };
    // 推导“最大可用 Z”（数组长度-1；索引0为Ghost）
    inline constexpr std::size_t BRAGG_MAX_Z =
        sizeof(BRAGG_RADII_ANG) / sizeof(double) - 1;

    // SG1 半径（单位 = bohr，本表只到 Ar；Z>18 时建议回退 BRAGG）
    inline constexpr double SG1_RADII_BOHR[] = {
        /*0*/ 0.0, /*1*/ 1.0000, /*2*/ 0.5882,
        /*3*/ 3.0769, 2.0513, 1.5385, 1.2308, 1.0256, 0.8791, 0.7692, 0.6838,
        /*11*/ 4.0909, 3.1579, 2.5714, 2.1687, 1.8750, 1.6514, 1.4754, 1.3333};
    inline constexpr std::size_t SG1_MAX_Z =
        sizeof(SG1_RADII_BOHR) / sizeof(double) - 1;
        
    // Treutler–Ahlrichs xi（无量纲；与你的列表一致）
    // double* xi_table 用于控制
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

    // ====== 安全取值函数（统一返回 bohr；带 fallback） ===========================
    //
    // 设计要点：
    // 1) BRAGG：Å → bohr 转换在函数内完成；索引 0 (Ghost) 使用 unknown_ang。
    // 2) SG1：只到 Ar，超范围自动回退 BRAGG。
    // 3) TA ξ：缺失时回退到 1.5（经验值）。
    // ---------------------------------------------------------------------------

    double bragg_radius_bohr(std::uint32_t Z);

    double sg1_radius_bohr(std::uint32_t Z);

    inline  double ta_xi(std::uint32_t Z, double fallback = 1.5)
    {
        const std::size_t n = sizeof(TA_XI) / sizeof(double);
        if (Z < n)
        {
            const double v = TA_XI[Z];
            return (v == v) ? v : fallback;
        }
        return fallback;
    }

    // ====== 默认网格数（径向/角向）与 Lebedev 阶-点数映射 =========================
    inline constexpr int RAD_GRIDS[10][7] = {
        /*period→       0    1    2    3    4    5    6   */
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
        /*period→       0    1    2    3    4    5    6   */
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

    // 元素周期分段阈值（Z）：(0..2], (3..10], (11..18], (19..36], (37..54], (55..86], (87..118]
    inline constexpr int PERIOD_THRESHOLDS[7] = {2, 10, 18, 36, 54, 86, 118};
    // ---- 工具：Z→周期索引（0..6），越界 clamp 到 6 ----
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

    // ---- 工具：level clamp 到 0..9 ----
    inline constexpr int clamp_level(int level)
    {
        return (level < 0 ? 0 : (level > 9 ? 9 : level));
    }
    // ====== 对外取值函数 ==========================================================
    // 默认径向壳层数（level: 0..9；默认 3）
    inline  int default_rad(unsigned Z, int level = 3)
    {
        const int L = clamp_level(level);
        const int P = period_index(Z);
        return RAD_GRIDS[L][P];
    }
    // 默认角向点数（Lebedev 阶数；level: 0..9；默认 3）
    inline  int default_ang(unsigned Z, int level = 3)
    {
        const int L = clamp_level(level);
        const int P = period_index(Z);
        return chemgrid::LEBEDEV_ORDER[ANG_ORDER[L][P]];
    }
    

} // namespace chemgrid::constants

namespace chemgrid
{

    // ================================
    // Enumeration Constants
    // ================================

    // ---- Radii Adjustment Methods ----
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

} // namespace chemgrid