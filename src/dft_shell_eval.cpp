// // shell_eval.cpp
#include "dft_shell_eval.hpp"

namespace chemgrid::gbs
{
    /**
     * @brief Wrapper for calculating the Gamma function
     * Uses the standard library's tgamma function
     */
    double gamma(double x)
    {
        return std::tgamma(x);
    }
    /**
     * @brief Calculates the Gaussian integral int_0^inf x^n exp(-alpha x^2) dx
     *
     * @param n The exponent
     * @param alpha The alpha parameter (exponent factor)
     * @return The value of the integral
     */
    double gaussian_int(int n, double alpha)
    {
        double n1 = (n + 1) * 0.5;
        return gamma(n1) / (2.0 * std::pow(alpha, n1));
    }
    /**
     * @brief Calculates the normalization factor for the radial part of a GTO
     *
     * For g = r^l * exp(-alpha * r^2)
     * The normalization factor ensures that int_0^inf g^2 * r^2 dr = 1
     *
     * @param l Angular momentum quantum number
     * @param expnt The exponent alpha
     * @return The normalization factor
     */
    static double gto_norm(int l, double expnt)
    {
        if (l < 0)
        {
            throw std::invalid_argument("Angular momentum l should be >= 0");
        }
        return 1.0 / std::sqrt(gaussian_int(l * 2 + 2, 2.0 * expnt));
    }
    /**
     * @brief Step 1: Normalize each primitive Gaussian function
     *
     * @param l Angular momentum quantum number
     * @param exps Exponent vector
     * @param coeffs Coefficient vector [nprim]
     */
    static void normalize_primitives(int l,
                                         const std::vector<double> &exps,
                                         std::vector<double> &coeffs)
    {
        size_t nprim = exps.size();

        // Calculate normalization factor for each primitive Gaussian
        std::vector<double> norms(nprim);
        for (size_t p = 0; p < nprim; ++p)
        {
            norms[p] = gto_norm(l, exps[p]);
        }

        // Apply the normalization factor to the coefficients: coeffs[p] *= norms[p]
        for (size_t p = 0; p < nprim; ++p)
        {

            coeffs[p] *= norms[p];
        }
    }
    /**
     * @brief Step 2: Normalize the contracted Gaussian function as a whole
     *
     * @param l Angular momentum quantum number
     * @param exps Exponent vector [nprim]
     * @param coeffs Coefficient vector [nprim], will be modified in-place
     */
    void normalize_contracted(int l,
                              const std::vector<double> &exps,
                              std::vector<double> &coeffs)
    {
        size_t nprim = exps.size();

        if (coeffs.size() != nprim)
        {
            throw std::invalid_argument("coeffs size must equal exps size");
        }

        // Calculate the overlap integral matrix ee[i][j] = gaussian_int(l*2+2, exps[i] + exps[j])
        std::vector<std::vector<double>> ee(nprim, std::vector<double>(nprim));
        for (size_t i = 0; i < nprim; ++i)
        {
            for (size_t j = 0; j < nprim; ++j)
            {
                ee[i][j] = gaussian_int(l * 2 + 2, exps[i] + exps[j]);
            }
        }

        // Calculate s^2 = sum_p sum_q c_p * ee[p][q] * c_q
        double s_squared = 0.0;
        for (size_t p = 0; p < nprim; ++p)
        {
            for (size_t q = 0; q < nprim; ++q)
            {
                s_squared += coeffs[p] * ee[p][q] * coeffs[q];
            }
        }

        // Normalization factor s = 1 / sqrt(s_squared)
        double s = 1.0 / std::sqrt(s_squared);

        // Apply the normalization factor to all coefficients
        for (size_t p = 0; p < nprim; ++p)
        {
            coeffs[p] *= s;
        }
    }
    /**
     * @brief Complete two-step normalization process
     *
     * @param l Angular momentum quantum number
     * @param exps Exponent vector [nprim]
     * @param coeffs Coefficient vector [nprim], input/output parameter, will be modified in-place
     */
    void normalize_basis(int l,
                         const std::vector<double> &exps,
                         std::vector<double> &coeffs)
    {
        // Step 1: Primitive Gaussian normalization
        normalize_primitives(l, exps, coeffs);

        // Step 2: Contracted Gaussian normalization
        normalize_contracted(l, exps, coeffs);
    }
    /**
     * @brief Generate all Cartesian components (lx, ly, lz) for a given angular momentum l
     *
     * Rule: lx + ly + lz = l
     * Order: lx decreases first, then ly decreases, lz increases
     *
     * Example:
     * l=0: (0,0,0)
     * l=1: (1,0,0), (0,1,0), (0,0,1)  -> px, py, pz
     * l=2: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)  -> dxx, dxy, dxz, dyy, dyz, dzz
     */
    std::vector<std::array<int, 3>> generate_cartesian_components(int l)
    {
        std::vector<std::array<int, 3>> components;

        for (int lx = l; lx >= 0; --lx)
        {
            for (int ly = l - lx; ly >= 0; --ly)
            {
                int lz = l - lx - ly;
                components.push_back({lx, ly, lz});
            }
        }

        return components;
    }
    /**
     * @brief Get unique atomic numbers.
     *
     */
    std::vector<int> unique_Z_c(const std::vector<int> &Z)
    {
        std::vector<int> uniq = Z;
        std::sort(uniq.begin(), uniq.end());       // Sort
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end()); // Deduplicate
        return uniq;
    }
    /**
     * @brief Get the element symbols to be processed.
     *
     */
    std::vector<std::string> mol_elements(std::vector<int> unique_Z)
    {
        std::vector<std::string> r;
        for (int z : unique_Z)
            r.push_back(Z_to_symbol(z));
        return r;
    }
    /*
     * @brief Convert string element symbol to atomic number (charge).
     */
    int symbol_to_Z(std::string symbol)
    {
        // 1. Normalize the input string
        // Ensure we don't operate on an empty string
        if (symbol.empty())
        {
            return 0; // Return 0 for unrecognized
        }

        // Convert the first character to uppercase
        symbol[0] = std::toupper(static_cast<unsigned char>(symbol[0]));

        // Convert the remaining characters to lowercase
        if (symbol.length() > 1)
        {
            std::transform(symbol.begin() + 1, symbol.end(), symbol.begin() + 1,
                           [](unsigned char c)
                           { return std::tolower(c); });
        }

        // 2. Define a static lookup table
        // 'static const' ensures this map is built only once and shared across all function calls
        static const std::map<std::string, int> element_map = {
            {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10}, {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30}, {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}, {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90}, {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {"Am", 95}, {"Cm", 96}, {"Bk", 97}, {"Cf", 98}, {"Es", 99}, {"Fm", 100}, {"Md", 101}, {"No", 102}, {"Lr", 103}, {"Rf", 104}, {"Db", 105}, {"Sg", 106}, {"Bh", 107}, {"Hs", 108}, {"Mt", 109}, {"Ds", 110}, {"Rg", 111}, {"Cn", 112}, {"Nh", 113}, {"Fl", 114}, {"Mc", 115}, {"Lv", 116}, {"Ts", 117}, {"Og", 118} // "D" (Deuterium) and "T" (Tritium) are isotopes of H, Z=1
            ,
            {"D", 1},
            {"T", 1} // Virtual particle / placeholder
            ,
            {"X", 0} // Sometimes X is used for a dummy atom, Z=0
        };

        // 3. Find and return the value
        // Use the .find() method
        auto it = element_map.find(symbol);

        if (it != element_map.end())
        {
            // Found
            return it->second; // Return the value (int) from the map
        }
        else
        {
            // Not found
            return 0; // In quantum chemistry, 0 is often used for dummy atoms or unrecognized atoms
        }
    }
    /**
     * @brief Generate the complete AO list from the normalized basis set
     */
    std::vector<AODesc> generate_ao_list(
        const std::map<int, std::vector<atom_AO>> &normed_bas,
        const std::vector<int> &charges,
        const std::vector<std::array<double, 3>> &atom_coords)
    {
        std::vector<AODesc> ao_list;

        // Check inputs
        if (charges.size() != atom_coords.size())
        {
            throw std::invalid_argument("charges and atom_coords size mismatch");
        }

        // Iterate over each atom
        for (size_t atom_idx = 0; atom_idx < charges.size(); ++atom_idx)
        {
            int charge = charges[atom_idx];

            // Find the basis set for this element
            auto it = normed_bas.find(charge);
            if (it == normed_bas.end())
            {
                throw std::runtime_error("No basis set found for element with charge " +
                                         std::to_string(charge));
            }

            const std::vector<atom_AO> &basis_shells = it->second;

            // Iterate over each shell of this atom
            for (const atom_AO &shell : basis_shells)
            {
                int l = shell.l;

                // Generate all Cartesian components for this shell
                std::vector<std::array<int, 3>> cart_components =
                    generate_cartesian_components(l);

                // Create an AODesc for each Cartesian component
                for (const auto &comp : cart_components)
                {
                    AODesc ao;
                    ao.atom = atom_idx;
                    ao.l = l;
                    ao.lx = comp[0];
                    ao.ly = comp[1];
                    ao.lz = comp[2];
                    ao.exps = shell.exps;
                    ao.coeffs = shell.coeffs;

                    ao_list.push_back(ao);
                }
            }
        }

        return ao_list;
    }

    std::map<int, std::vector<atom_AO>> get_normed_ao(std::string gbs, std::vector<int> charges)
    {
        std::vector<int> u_z = chemgrid::gbs::unique_Z_c(charges);
        gansu::BasisSet bs_;
        std::ifstream fin(gbs);
        if (!fin)
        {
            throw std::runtime_error("Cannot open gbs: ");
        }
        // Store basis set
        std::vector<AODesc> aos;
        int atom_index = 0; // Atom index
        std::map<int, std::vector<atom_AO>> normed_bas;
        // Read basis set
        gansu::BasisSet bs = bs_.construct_from_gbs(gbs);
        std::vector<std::string> ELEMENT = chemgrid::gbs::mol_elements(u_z);

        for (std::string e : ELEMENT)
        {
            gansu::ElementBasisSet HB = bs.get_element_basis_set(e);
            int e_charge = chemgrid::gbs::symbol_to_Z(e);
            std::vector<atom_AO> e_AO;
            // std::cout << "element: " << e << std::endl;
            for (auto a : HB.get_contracted_gaussians())
            {
                std::string type = a.get_type();
                // std::cout << "type: " << type << "\n";
                int l = l_from_shell_char(type[0]);
                std::vector<gansu::PrimitiveGauss> primitives = a.get_primitives();
                std::vector<double> esp;
                std::vector<double> c;
                for (gansu::PrimitiveGauss b : primitives)
                {
                    // std::cout << "esp: " << b.exponent << "  c: " << b.coefficient << "\n";
                    esp.push_back(b.exponent);
                    c.push_back(b.coefficient);
                }
                // Process (Normalize)
                chemgrid::gbs::normalize_basis(l, esp, c);
                atom_AO ao;
                ao.l = l;
                ao.exps = esp;
                ao.coeffs = c;
                e_AO.push_back(ao);
            }
            normed_bas[e_charge] = e_AO;
        }
        return normed_bas;
    }

    void evaluate_AO_on_grid_screened(
        const std::vector<AODesc> &aos,
        const std::vector<std::array<double, 3>> &atm_coords,
        const std::vector<std::array<double, 3>> &grids,
        int blksize,
        const std::vector<uint8_t> &screen_index,
        std::vector<double> &ao_out)
    {
        const int nao = (int)aos.size();
        const int G = (int)grids.size();
        const int nblk = (G + blksize - 1) / blksize;

        if ((int)screen_index.size() != nblk * nao)
            throw std::runtime_error("evaluate_AO_on_grid_screened: screen_index size mismatch");

        ao_out.assign((size_t)nao * G, 0.0);

        // ---------- Skip statistics ----------
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
                    // Note: This 'continue' skips the entire block for this AO.
                    // We update statistics based on this block skip.
                    skipped_blocks++; 
                    skipped_points += dg;
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
                    if (ao.lx)
                        ang *= std::pow(rx, ao.lx);
                    if (ao.ly)
                        ang *= std::pow(ry, ao.ly);
                    if (ao.lz)
                        ang *= std::pow(rz, ao.lz);

                    ao_out[(size_t)mu * G + g] = ang * radial;
                }
            }
        }

        // ---------- Print statistics ----------
        double ratio_blocks = (total_blocks > 0) ? (100.0 * skipped_blocks / (double)total_blocks) : 0.0;
        double ratio_points = (G > 0 && nao > 0) ? (100.0 * skipped_points / (double)(G * nao)) : 0.0;


        printf("[screened AO eval] skipped blocks: %zu / %zu (%.2f%%)\n",
               skipped_blocks, total_blocks, ratio_blocks);
        printf("[screened AO eval] approx skipped points: %zu / %zu (%.2f%%)\n",
               skipped_points, (size_t)G * nao, ratio_points);
    }

} // namespace chemgrid::gbs