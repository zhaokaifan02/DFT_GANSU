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
#include "dft_genGrid.hpp"
using namespace std;


int main(int argc, char **argv)
{
    // ============ 新增：从参数决定输入/输出前缀 ============
    // 例：argv[1] = "DHA" -> 读 "DHA.txt"，写 "DHA_grid.txt"
    std::string tag = (argc >= 2 && argv[1] && std::string(argv[1]).size() > 0)
                          ? std::string(argv[1])
                          : std::string("mol");
    chemgrid::gen_grid_screen(tag);
    // const std::string input_file = tag + ".txt";
    // const std::string final_out = tag + "_grid.txt";
    // // ======================================================

    // std::vector<double> x, y, z, w;
    // std::vector<int> atom_id;

    // // 分子data
    // std::vector<int> charges;
    // std::vector<std::array<double, 3>> atm_coords;
    // read_mol_txt(input_file, charges, atm_coords);

    // std::vector<int> atom_nuc = unique_Z(charges);

    // // 原子GRID
    // auto start1 = std::chrono::high_resolution_clock::now();
    // chemgrid::grid::gen_atom_grid(x, y, z, w, atom_id, atom_nuc,
    //                               chemgrid::RadialMethod::TreutlerAhlrichs, 3,
    //                               chemgrid::PruningMethod::NWChem);
    // auto end1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration1 = end1 - start1;
    // std::cout << "gen_atom: " << duration1.count() << " seconds" << std::endl;

    // auto grouped = group_by_atom(x, y, z, w, atom_id);

    // for (const auto &kv : grouped)
    // {
    //     int atom = kv.first;
    //     const auto &points = kv.second;
    //     std::cout << "Atom " << atom << " has " << points.size() << " points\n";
    //     if (!points.empty())
    //     {
    //         std::cout << " First point = (" << points[0][0] << ", "
    //                   << points[0][1] << ", " << points[0][2]
    //                   << "), w = " << points[0][3] << "\n";
    //         // 可选：把每个原子的原子网格也带上 tag 前缀（不要求，但更清晰）
    //         // export_grids_txt0(tag + "_atom_" + std::to_string(atom) + "_grid.txt", points);
    //     }
    // }

    // chemgrid::grid::PartitionOut part_h2o;
    // // partition 部分
    // auto start2 = std::chrono::high_resolution_clock::now();
    // // 原子坐标，grouped
    // part_h2o = chemgrid::grid::get_partition_from_templates_GPU_warp_ownerOnly(
    //     atm_coords, grouped, charges,
    //     chemgrid::constants::BRAGG_RADII_ANG); // 8 streams
    // auto end2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration2 = end2 - start2;
    // std::cout << "partition: " << duration2.count() << " seconds" << std::endl;

    // std::cout << tag << " partition grid has " << part_h2o.first.size() << " points\n";

    // // 全局化
    // std::vector<int> atm_idx;               // 输出
    // std::vector<double> quadrature_weights; // 输出
    // auto start3 = std::chrono::high_resolution_clock::now();
    // // 填充+重排

    // chemgrid::grid::build_atm_idx_and_weights_by_Z(grouped, charges, part_h2o.first, part_h2o.second, atm_idx, quadrature_weights);

    // // export_grid2_txt("h2o_global.txt", part_h2o.first, quadrature_weights, atm_idx);
    // auto end3 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration3 = end3 - start3;
    // std::cout << "build_atm_idx_and_weights_by_Z: " << duration3.count() << " seconds" << std::endl;
    // // // 重排
    // auto start4 = std::chrono::high_resolution_clock::now();
    // std::vector<size_t> idx = chemgrid::grid::arg_group_grids_cpu(part_h2o.first, atm_coords,
    //                                                               chemgrid::constants::GROUP_BOX_SIZE,          // 1.2
    //                                                               chemgrid::constants::GROUP_BOUNDARY_PENALTY); // 4.2
    // chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.first);
    // chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.second);
    // chemgrid::grid::reorder_by_index_inplace(idx, atm_idx);
    // chemgrid::grid::reorder_by_index_inplace(idx, quadrature_weights);
    // auto end4 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration4 = end4 - start4;
    // std::cout << "reorder: " << duration4.count() << " seconds" << std::endl;
    // // 填充
    // auto start5 = std::chrono::high_resolution_clock::now();
    // chemgrid::grid::pad_grids_cpp(part_h2o.first, part_h2o.second, atm_idx, quadrature_weights, chemgrid::constants::ALIGNMENT_UNIT);
    // auto end5 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration5 = end5 - start5;
    // std::cout << "pad: " << duration5.count() << " seconds" << std::endl;

    // // ============ 修改：最终输出文件名用 tag ============
    // export_grid3_txt(final_out, part_h2o.first, part_h2o.second, quadrature_weights, atm_idx);
    // // =====================================================

    // std::cout << "total time: " << (duration1.count() + duration2.count() + duration3.count() + duration5.count())
    //           << " seconds" << std::endl;
    // gansu::BasisSet bs_;
    // std::ifstream fin("./basis/sto-3g.gbs");
    // if (!fin)
    // {
    //     throw std::runtime_error("Cannot open gbs: ");
    // }

    // gansu::BasisSet bs = bs_.construct_from_gbs("./basis/sto-3g.gbs");
    // auto aos = build_ao_descs_from_mol_and_gbs(charges, atm_coords, bs, {/*normalize_cart*/ true});
    // int nao = (int)aos.size();
    // int G = (int)part_h2o.first.size();
    // std::vector<double> AO; // 大小 = nao * G
    // evaluate_AO_on_grid(aos, atm_coords, part_h2o.first, AO);
    // write_ao_ngrid_nao_txt(tag + "_AO.txt", AO.data(), G, nao);
    // std::cout << "AO shape: (" << G << ", " << nao << ")\n";
    return 0;
}
