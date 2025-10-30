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
#include "dft_genGrid.hpp"
#include "dft_aoEVAL.hpp"
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
        throw std::runtime_error("无法打开文件: " + path);
    }

    fout << ngrids << " " << nao << "\n";
    fout << std::scientific << std::setprecision(17);

    // 按行主序写入：ao_out[g * nao + mu]
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
void export_GRID_txt(const std::string &filename,
                     const std::vector<std::array<double, 3>> &coords,
                     const std::vector<double> &weights,
                     const std::vector<double> &quadrature_weights,
                     const std::vector<int> &atm_idx)
{
    const std::size_t N = coords.size();
    std::cout << "N=" << N << ", weights.size()=" << weights.size()
              << ", quadrature_weights.size()=" << quadrature_weights.size()
              << ", atm_idx.size()=" << atm_idx.size() << std::endl;

    if (weights.size() != N || atm_idx.size() != N)
        throw std::invalid_argument("export_grid_txt: 输入数组大小不一致");

    std::ofstream ofs(filename);
    if (!ofs)
        throw std::runtime_error("无法打开输出文件: " + filename);

    ofs << std::scientific << std::setprecision(20);
    for (std::size_t i = 0; i < N; ++i)
    {
        ofs << atm_idx[i] << " "
            << coords[i][0] << " "
            << coords[i][1] << " "
            << coords[i][2] << " "
            << weights[i] << " "
            << quadrature_weights[i] << "\n";
    }
    ofs.close();
}

// struct AODesc
// {
//     int atom;                   // 该 AO 所在的原子索引
//     int l;                      // 壳角动量
//     int lx, ly, lz;             // 笛卡尔幂次数
//     std::vector<double> exps;   // α_j
//     std::vector<double> coeffs; // c_j（含或不含归一化，视你的设定）
// };

int main(int argc, char **argv)
{
    // ============ 新增：从参数决定输入/输出前缀 ============
    // 例：argv[1] = "DHA" -> 读 "DHA.txt"，写 "DHA_grid.txt"
    std::string tag = (argc >= 2 && argv[1] && std::string(argv[1]).size() > 0)
                          ? std::string(argv[1])
                          : std::string("mol");
    std::string gbs_tag = (argc >= 3 && argv[2] && std::string(argv[2]).size() > 0)
                              ? std::string(argv[2])
                              : std::string("sto-3g");

    const std::string final_out = tag + "_grid.txt";
    const std::string final_outAO = tag + "_AO.txt";
    std::vector<int> charges;
    std::vector<std::array<double, 3>> atm_coords;
    chemgrid::dft_getMolfromTXT(tag, charges, atm_coords);
    std::vector<int> u_z = chemgrid::gbs::unique_Z_c(charges);
    // 生成网格
    std::pair<std::vector<std::array<double, 3>>, std::vector<double>> grids = chemgrid::dft_gen_grid(charges, atm_coords);
    std::vector<int> atm_idx(grids.first.size(), 0); // 占位符
    export_GRID_txt(final_out, grids.first, grids.second, grids.second, atm_idx);
    std::map<int, std::vector<atom_AO>> normed_bas = chemgrid::gbs::get_normed_ao("./" + gbs_tag + ".gbs", charges);
    // // 读取基组
    // gansu::BasisSet bs = bs_.construct_from_gbs("./cc-pvdz.gbs");
    // std::vector<std::string> ELEMENT = chemgrid::gbs::mol_elements(u_z);

    // for (std::string e : ELEMENT)
    // {
    //     gansu::ElementBasisSet HB = bs.get_element_basis_set(e);
    //     int e_charge = chemgrid::gbs::symbol_to_Z(e);
    //     std::vector<atom_AO> e_AO;
    //     std::cout << "element: " << e << std::endl;
    //     for (auto a : HB.get_contracted_gaussians())
    //     {
    //         std::string type = a.get_type();
    //         std::cout << "type: " << type << "\n";
    //         int l = shell_char_to_l(type[0]);
    //         std::vector<gansu::PrimitiveGauss> primitives = a.get_primitives();
    //         std::vector<double> esp;
    //         std::vector<double> c;
    //         for (gansu::PrimitiveGauss b : primitives)
    //         {
    //             std::cout << "esp: " << b.exponent << "  c: " << b.coefficient << "\n";
    //             esp.push_back(b.exponent);
    //             c.push_back(b.coefficient);
    //         }
    //         // 处理
    //         chemgrid::gbs::normalize_basis(l, esp, c);
    //         atom_AO ao;
    //         ao.l = l;
    //         ao.exps = esp;
    //         ao.coeffs = c;
    //         e_AO.push_back(ao);
    //     }
    //     normed_bas[e_charge] = e_AO;
    // }

    // 遍历处理后的各个轨道
    // for (auto pair : normed_bas)
    // {
    //     int charge = pair.first;
    //     std::string symbol = chemgrid::gbs::Z_to_symbol(charge);
    //     std::cout << "ELEMENT: " << symbol << std::endl;

    //     std::vector<atom_AO> aos = pair.second;
    //     for (atom_AO ao : aos)
    //     {
    //         std::cout << "type(l): " << ao.l << "\n";
    //         for (int i = 0; i < ao.exps.size(); ++i)
    //         {
    //             std::cout << "esp: " << ao.exps[i] << "  c: " << ao.coeffs[i] << "\n";
    //         }
    //     }
    // }
    std::vector<AODesc> AODESC = chemgrid::gbs::generate_ao_list(normed_bas, charges, atm_coords);
    int ngrids = static_cast<int>(grids.first.size());
    int nao = static_cast<int>(AODESC.size());
    double *ao_values = new double[ngrids * nao];
    chemgrid::AOEval::evaluate_aos_on_grids_gpu_raw(AODESC,atm_coords,grids.first,ao_values,ngrids,nao);
    write_ao_txt(final_outAO, ao_values, ngrids, nao);
    std::cout << "AO written to ao.txt (" << ngrids << " x " << nao << ")" << std::endl;
    // chemgrid::gbs::GBSReader reader;
    // reader.build("./cc-pvdz.gbs");
    // cout << "Loaded elements: \n";
    // // 查看 H 和 O 的基组信息
    // chemgrid::gbs::ElementBasis HB = reader.element_basis("H");
    // chemgrid::gbs::ElementBasis OB = reader.element_basis("O");
    // for (const auto &shell : HB.shells)
    // {
    //     cout << "H shell type: " << shell.type << ", l: " << shell.l << ", nprim: " << shell.exps.size() << endl;
    // }
    // cout << "OB shells:\n";
    // cout << OB.shells.size() << " shells found for O element.\n";
    // cout << HB.shells.size() << " shells found for H element.\n";
    // for (const auto &shell : OB.shells)
    // {
    //     cout << "O shell type: " << shell.type << ", l: " << shell.l << ", nprim: " << shell.exps.size() << endl;
    // }
    // // 生成 AO 列表
    // std::cout << "Generated grid for tag: " << tag << endl;
    // std::vector<AODesc> aos = chemgrid::gbs::build_aodescs_from_gbs(reader, charges, atm_coords);
    // cout << "Generated grid for tag: " << tag << endl;
    // std::cout << "AOS size: " << aos.size() << std::endl;
    // std::vector<double> ao_out;
    // chemgrid::AOEval::evaluate_AO_on_grid_GPU_AOmajor(aos, atm_coords, grids.first, ao_out);
    // std::cout << "Evaluated AO on grid, ao_out size: " << ao_out.size() << std::endl;
    // int ngrids = static_cast<int>(grids.first.size());
    // int nao = static_cast<int>(aos.size());

    // write_ao_txt(final_outAO, ao_out, ngrids, nao);
    // std::cout << "AO written to ao.txt (" << ngrids << " x " << nao << ")" << std::endl;

    // 写 AO 结果
    return 0;
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
