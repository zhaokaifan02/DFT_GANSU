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
#include "dft_genGrid.hpp"
#include "dft_aoEVAL.hpp"
#include "dft_atomGrid.hpp"
#include <map>
using namespace std;
void write_ao_txt(const std::string &path,
                  const double *ao_out,
                  int ngrids,
                  int nao)
{
    std::ofstream fout(path);
    if (!fout.is_open())
    {
        throw std::runtime_error("open file: " + path);
    }

    fout << ngrids << " " << nao << "\n";
    fout << std::scientific << std::setprecision(17);

    // ao_out[g * nao + mu]
    for (int g = 0; g < ngrids; ++g)
    {
        for (int mu = 0; mu < nao; ++mu)
        {
            fout << std::setw(25) << ao_out[g * nao + mu];
            if (mu + 1 < nao)
                fout << " ";
        }
        fout << "\n";
    }

    fout.close();
}
void write_GRID_txt(const std::string &filename,
                     const std::vector<std::array<double, 3>> &coords,
                     const std::vector<double> &weights)
{
    const std::size_t N = coords.size();
    std::cout << "N=" << N << ", weights.size()=" << weights.size()
              <<  std::endl;

    std::ofstream ofs(filename);
    if (!ofs)
        throw std::runtime_error("error file : " + filename);

    ofs << std::scientific << std::setprecision(20);
    for (std::size_t i = 0; i < N; ++i)
    {
        ofs << coords[i][0] << " "
            << coords[i][1] << " "
            << coords[i][2] << " "
            << weights[i] <<"\n";
    }
    ofs.close();
}

int main(int argc, char **argv)
{
    // ：argv[1] = "DHA" ->  "DHA.txt"， "DHA_grid.txt"
    std::string tag = (argc >= 2 && argv[1] && std::string(argv[1]).size() > 0)
                          ? std::string(argv[1])
                          : std::string("mol");
    std::string gbs_tag = (argc >= 3 && argv[2] && std::string(argv[2]).size() > 0)
                              ? std::string(argv[2])
                              : std::string("sto-3g");
    // create mol
    std::vector<int> charges;
    std::vector<std::array<double, 3>> atm_coords;

    chemgrid::dft_getMolfromTXT(tag, charges, atm_coords);
    // get coords weight 
    std::pair<std::vector<std::array<double, 3>>,std::vector<double>> grids = chemgrid::dft_gen_grid(charges,atm_coords);

    //get AO grids (ngrids,naos)
    //create basis set
    gansu::BasisSet bs_;
    gansu::BasisSet bs = bs_.construct_from_gbs("./basis/"+gbs_tag+".gbs");
    chemgrid::AOGrids aoGrids = chemgrid::dft_gen_ao(bs,charges,atm_coords,grids.first);

    //output txt for LDA.py
    const std::string grids_output_file = "./grids/"+tag+"_grid.txt";
    const std::string aos_output_file = "./grids/" +tag+"_AO.txt";
    write_ao_txt(aos_output_file,aoGrids.ao,aoGrids.ngrids,aoGrids.naos);
    write_GRID_txt(grids_output_file,grids.first,grids.second);
    return 0;
}
