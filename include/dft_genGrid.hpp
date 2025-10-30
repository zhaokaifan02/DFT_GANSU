#include "dft_atomGrid.hpp"
#include <iostream>
#include <array>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <string>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include "dft_shell_eval.hpp"
#include "basis_set.hpp"
#include "types.hpp"

namespace chemgrid
{
    struct AOGrids
    {
        double *ao;
        int ngrids;
        int naos;
    };
    void gen_grid(std::string tag, std::string basis);
    void gen_grid_screen(std::string tag, std::string basis);
    std::pair<std::vector<std::array<double, 3>>, std::vector<double>> dft_gen_grid(std::string tag);
    /** 
     *  @brief get grids 
     *  
     *  ouput {coords(x,y,z), weights(w)}
     */
    std::pair<std::vector<std::array<double, 3>>, std::vector<double>> dft_gen_grid(std::vector<int> charges, std::vector<std::array<double, 3>> atm_coords);
    void dft_getMolfromTXT(std::string tag, std::vector<int> &charges, std::vector<std::array<double, 3>> &atm_coords);
    std::map<int, std::vector<atom_AO>> dft_get_normed_basis(gansu::BasisSet bs, std::vector<int> charges);

    /**
     * @brief get ao grids
     * output (ngrids, naos)
     *
     */
    AOGrids dft_gen_ao(gansu::BasisSet &bs, std::vector<int> charges, std::vector<std::array<double, 3>> &atm_coords, std::vector<std::array<double, 3>> coords);
}
