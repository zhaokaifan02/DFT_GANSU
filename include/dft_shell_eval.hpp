// shell_eval.hpp
#pragma once
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "types.hpp"     // Atom, Coordinate, AngularMomentums, angstrom_to_bohr
#include "basis_set.hpp" // BasisSet, ElementBasisSet, ContractedGauss
#include <map>
// ====================== AO Descriptor: One record represents one AO component ======================
struct AODesc
{
    int atom;                   // Index of the atom this AO belongs to
    int l;                      // Shell angular momentum
    int lx, ly, lz;             // Cartesian exponents
    std::vector<double> exps;   // α_j (exponents)
    std::vector<double> coeffs; // c_j (normalized)
};
struct atom_AO
{
    int l;
    std::vector<double> exps;
    std::vector<double> coeffs; // Processed (e.g., normalized)
};

namespace chemgrid::gbs
{

    // ============ Basic Data Structures ============

    // A "Shell" (SP has been split into separate S or P)
    struct Shell
    {
        char type;                  // Shell type: 'S','P','D','F','G'...
        int l;                      // Angular momentum: 0, 1, 2, 3,...
        std::vector<double> exps;   // α_j
        std::vector<double> coeffs; // c_j (raw coefficients, not yet normalized for primitives/components)
    };
    // ================= Utility: Z -> Element Symbol =================
    static std::string Z_to_symbol(int Z)
    {
        static const char *tbl[] = {
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr"};
        if (Z >= 1 && Z <= (int)(sizeof(tbl) / sizeof(tbl[0])))
            return tbl[Z - 1];
        throw std::runtime_error("Z_to_symbol: unsupported Z=" + std::to_string(Z));
    }
    // Element Basis Set: multiple shells under the same element
    inline int l_from_shell_char(char c)
    {
        switch (c)
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
        default:
            throw std::runtime_error(std::string("Unsupported shell type: ") + c);
        }
    }
    /**
     * @brief Get unique atomic numbers (Z) from the molecule.
     *
     */
    std::vector<int> unique_Z_c(const std::vector<int> &Z);
    /**
     * @brief Get element symbols for the molecule.
     *
     */
    std::vector<std::string> mol_elements(std::vector<int> unique_Z);
    /*
     * @brief Convert string element symbol to atomic number (Z).
     */
    int symbol_to_Z(std::string symbol);

    /**
     * @brief Complete two-step normalization process
     *
     * @param l Angular momentum quantum number
     * @param exps Exponent vector [nprim]
     * @param coeffs Coefficient vector [nprim], input/output parameter, will be modified in-place
     */
    void normalize_basis(int l,
                         const std::vector<double> &exps,
                         std::vector<double> &coeffs);
    inline std::vector<uint8_t> build_screen_index_no_omp(
        const std::vector<AODesc> &aos,
        const std::vector<std::array<double, 3>> &atm_coords, // Bohr
        const std::vector<std::array<double, 3>> &grid,       // AoS (G×3)
        int blksize = 56,                                     // Recommended: 64/128/256
        int nbins = 100,                                      // Recommended: 32~128
        double cutoff = 1e-15                                 // >0, and not too unreasonable
    )
    {
        // ---- Basic validity checks ----
        if (grid.empty() || aos.empty())
            return {};
        if (blksize <= 0)
            throw std::runtime_error("build_screen_index_no_omp: blksize must be > 0");
        if (!(cutoff > 0.0))
            throw std::runtime_error("build_screen_index_no_omp: cutoff must be > 0");

        const int G = (int)grid.size();
        const int nAO = (int)aos.size();
        const int nblk = (G + blksize - 1) / blksize;

        nbins = std::min(std::max(nbins, 1), 127);

        // Prevent log(<=0)
        const double cutoff_c = std::min(cutoff, 0.1);
        if (!(cutoff_c > 0.0))
            throw std::runtime_error("build_screen_index_no_omp: cutoff too small");

        const double scale = -(double)nbins / std::log(cutoff_c);

        std::vector<uint8_t> screen_index((size_t)nblk * (size_t)nAO, 0);
        std::vector<double> rr_buf((size_t)blksize);

        for (int ao_id = 0; ao_id < nAO; ++ao_id)
        {
            const AODesc &ao = aos[ao_id];
            const int l = ao.l;

            // Atom index and data validity
            if (ao.atom < 0 || ao.atom >= (int)atm_coords.size())
                continue; // Entire column is 0
            if (ao.exps.empty() || ao.coeffs.empty())
                continue; // Entire column is 0

            // Optional: If you require strictly equal lengths, enable the line below
            // if (ao.exps.size() != ao.coeffs.size()) continue;

            const double ax = atm_coords[ao.atom][0];
            const double ay = atm_coords[ao.atom][1];
            const double az = atm_coords[ao.atom][2];

            double min_exp = 1e300;
            for (double a : ao.exps)
                min_exp = std::min(min_exp, a);

            double maxc = 0.0;
            for (double c : ao.coeffs)
                maxc = std::max(maxc, std::fabs(c));

            if (!(min_exp > 0.0) || !(maxc > 0.0))
                continue; // Avoid log(0) or negative exponents

            const double log_coeff = std::log(maxc);

            // Inner region boundary for l > 0
            double r2sup = 0.0;
            double a_rr_min = -log_coeff;
            if (l > 0)
            {
                r2sup = (double)l / (2.0 * min_exp);
                if (r2sup <= 0.0 || !std::isfinite(r2sup))
                    continue;
                a_rr_min = min_exp * r2sup - 0.5 * std::log(r2sup) * l - log_coeff;
            }

            for (int ib = 0; ib < nblk; ++ib)
            {
                const int g0 = ib * blksize;
                const int g1 = std::min(G, (ib + 1) * blksize);
                const int dg = g1 - g0;
                if (dg <= 0)
                    continue;

                for (int i = 0; i < dg; ++i)
                {
                    const double dx = grid[g0 + i][0] - ax;
                    const double dy = grid[g0 + i][1] - ay;
                    const double dz = grid[g0 + i][2] - az;
                    rr_buf[(size_t)i] = dx * dx + dy * dy + dz * dz;
                }

                double rr_min = 1e300;
                for (int i = 0; i < dg; ++i)
                    rr_min = std::min(rr_min, rr_buf[(size_t)i]);

                double a_rr;
                if (l == 0)
                {
                    a_rr = min_exp * rr_min - log_coeff;
                }
                else if (rr_min < r2sup)
                {
                    a_rr = a_rr_min; // Avoid log(0)
                }
                else
                {
                    // Only reached when rr_min > 0: safe
                    a_rr = min_exp * rr_min - 0.5 * std::log(rr_min) * l - log_coeff;
                }

                // Map to discrete level (robustly handle NaN/Inf)
                uint8_t si_byte = 0;
                if (std::isfinite(a_rr))
                {
                    const double si_real = (double)nbins - a_rr * scale;
                    if (si_real > 0.0 && std::isfinite(si_real))
                    {
                        const int si = (int)(si_real + 1.0);
                        si_byte = (uint8_t)std::min(si, nbins);
                    }
                }
                screen_index[(size_t)ib * (size_t)nAO + (size_t)ao_id] = si_byte;
            }
        }
        return screen_index;
    }

} // namespace gbs

namespace chemgrid::gbs
{

    std::map<int, std::vector<atom_AO>> get_normed_ao(std::string gbs, std::vector<int> charges);
    std::vector<AODesc> generate_ao_list(
        const std::map<int, std::vector<atom_AO>> &normed_bas,
        const std::vector<int> &charges,
        const std::vector<std::array<double, 3>> &atom_coords);

    void evaluate_AO_on_grid_screened(
        const std::vector<AODesc> &aos,                       // nao
        const std::vector<std::array<double, 3>> &atm_coords, // A×3 (Bohr)
        const std::vector<std::array<double, 3>> &grids,      // G×3 (AoS)
        int blksize,                                          // Must be consistent with the one used for building screen_index
        const std::vector<uint8_t> &screen_index,             // Size: nblk*nao
        std::vector<double> &ao_out                           // Output: nao*G
    );
}