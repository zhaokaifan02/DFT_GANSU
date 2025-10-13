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
}
