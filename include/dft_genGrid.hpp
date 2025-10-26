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
#include "shell_eval.hpp"
#include "basis_set.hpp"
#include "types.hpp"

namespace chemgrid {
    void gen_grid(std::string tag);
    void gen_grid_screen(std::string tag);
    std::pair<std::vector<std::array<double, 3>>,std::vector<double>> dft_gen_grid(std::string tag); //输出原子坐标的接口
    std::pair<std::vector<std::array<double, 3>>,std::vector<double>> dft_gen_grid(std::vector<int> charges,std::vector<std::array<double, 3>> atm_coords);
    void dft_getMolfromTXT(std::string tag, std::vector<int> &charges, std::vector<std::array<double, 3>> &atm_coords);
}   
