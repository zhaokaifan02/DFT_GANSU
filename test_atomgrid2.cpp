#include "dft_atomGrid.hpp"
#include <iostream>
#include <array>
#include <fstream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <string>
#include <chrono>
std::vector<int> unique_Z(const std::vector<int>& Z) {
    std::vector<int> uniq = Z;
    std::sort(uniq.begin(), uniq.end());            // 排序
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end()); // 去重
    return uniq;
}
void read_mol_txt(const std::string &filename,
                  std::vector<int> &Z,
                  std::vector<std::array<double, 3>> &coords)
{
    std::ifstream ifs(filename);
    if (!ifs)
        throw std::runtime_error("Cannot open file " + filename);

    size_t n;
    ifs >> n;
    std::string line;
    std::getline(ifs, line); // 跳过行尾
    std::getline(ifs, line); // 注释行

    Z.resize(n);
    coords.resize(n);

    for (size_t i = 0; i < n; i++)
    {
        int z;
        double x, y, zc;
        ifs >> z >> x >> y >> zc;
        Z[i] = z;
        coords[i] = {x, y, zc};
    }
}
void export_per_atom_txt(
    const std::string &out_dir,
    const std::vector<std::array<double, 3>> &coords, // N×3
    const std::vector<double> &weights,               // N
    const std::vector<int> &atm_idx,                  // N, 0..natm-1
    const std::vector<int> &charges_Z                 // natm, 可空：若为空则不在文件名写Z
)
{
    const std::size_t N = coords.size();
    if (weights.size() != N || atm_idx.size() != N)
        throw std::invalid_argument("export_per_atom_txt: size mismatch");

    // 创建目录
    std::filesystem::create_directories(out_dir);

    // 找到 natm
    int natm = 0;
    for (int ia : atm_idx)
        natm = std::max(natm, ia + 1);

    // 为每个原子准备一个 ofstream
    std::vector<std::ofstream> files(natm);
    for (int ia = 0; ia < natm; ++ia)
    {
        std::string fname;
        if (!charges_Z.empty() && ia < (int)charges_Z.size())
        {
            int Z = charges_Z[ia];
            char buf[256];
            std::snprintf(buf, sizeof(buf), "atom_%03d_Z%02d.txt", ia, Z);
            fname = (std::filesystem::path(out_dir) / buf).string();
        }
        else
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "atom_%03d.txt", ia);
            fname = (std::filesystem::path(out_dir) / buf).string();
        }
        files[ia].open(fname);
        if (!files[ia])
            throw std::runtime_error("Cannot open " + fname);
        files[ia] << std::scientific << std::setprecision(12);
    }

    // 写数据：按原子分发
    for (std::size_t i = 0; i < N; ++i)
    {
        int ia = atm_idx[i];
        if (ia < 0 || ia >= natm)
            continue; // 填充点或非法，跳过
        files[ia] << coords[i][0] << " "
                  << coords[i][1] << " "
                  << coords[i][2] << " "
                  << weights[i] << "\n";
    }

    // 关闭
    for (auto &f : files)
        f.close();
}
void export_grids_txt0(const std::string &filename,
                       const std::vector<std::array<double, 4>> &grapds)

{
    std::ofstream ofs(filename);
    if (!ofs)
        throw std::runtime_error("Cannot open file for writing");
    ofs << std::fixed << std::setprecision(20);
    for (std::size_t i = 0; i < grapds.size(); ++i)
    {
        ofs << grapds[i][0] << " "
            << grapds[i][1] << " "
            << grapds[i][2] << " "
            << grapds[i][3] << "\n";
    }
}
void export_grids_txt(const std::string &filename,
                      const std::vector<std::array<double, 3>> &coords,
                      const std::vector<double> &weights)
{
    std::ofstream ofs(filename);
    if (!ofs)
        throw std::runtime_error("Cannot open file for writing");
    ofs << std::fixed << std::setprecision(20);
    for (std::size_t i = 0; i < coords.size(); ++i)
    {
        ofs << coords[i][0] << " "
            << coords[i][1] << " "
            << coords[i][2] << " "
            << weights[i] << "\n";
    }
}

void export_grid2_txt(const std::string &filename,
                      const std::vector<std::array<double, 3>> &coords,
                      const std::vector<double> &weights,
                      const std::vector<int> &atm_idx)
{
    const std::size_t N = coords.size();
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
            << weights[i] << "\n";
    }
    ofs.close();
}
void export_grid3_txt(const std::string &filename,
                      const std::vector<std::array<double, 3>> &coords,
                      const std::vector<double> &weights,
                      const std::vector<double> &quadrature_weights,
                      const std::vector<int> &atm_idx)
{
    const std::size_t N = coords.size();
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
std::unordered_map<int, std::vector<std::array<double, 4>>>
group_by_atom(const std::vector<double> &x,
              const std::vector<double> &y,
              const std::vector<double> &z,
              const std::vector<double> &w,
              const std::vector<int> &atom_id)
{
    if (x.size() != y.size() || x.size() != z.size() ||
        x.size() != w.size() || x.size() != atom_id.size())
    {
        throw std::runtime_error("group_by_atom: input vectors have inconsistent sizes");
    }

    std::unordered_map<int, std::vector<std::array<double, 4>>> result;

    const std::size_t N = x.size();
    for (std::size_t i = 0; i < N; ++i)
    {
        int atom = atom_id[i];
        std::array<double, 4> point = {x[i], y[i], z[i], w[i]};
        result[atom].push_back(point);
    }

    return result;
}
int main()
{
    std::vector<double> x, y, z, w;
    std::vector<int> atom_id;
    

    //分子data
    std::vector<int> charges;
    std::vector<std::array<double, 3>> atm_coords;
    read_mol_txt("mol.txt", charges, atm_coords);

    
    std::vector<int> atom_nuc = unique_Z(charges);

    // 原子GRID
    auto start1 = std::chrono::high_resolution_clock::now();
    chemgrid::grid::gen_atom_grid(x, y, z, w, atom_id, atom_nuc, chemgrid::RadialMethod::TreutlerAhlrichs, 3, chemgrid::PruningMethod::NWChem);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start1;
    std::cout << "gen_atom: " << duration1.count() << " seconds" << std::endl;

    auto grouped = group_by_atom(x, y, z, w, atom_id);

    for (const auto &kv : grouped)
    {
        int atom = kv.first;
        const auto &points = kv.second;
        std::cout << "Atom " << atom << " has " << points.size() << " points\n";
        if (!points.empty())
        {
            std::cout << " First point = (" << points[0][0] << ", "
                      << points[0][1] << ", " << points[0][2]
                      << "), w = " << points[0][3] << "\n";
            export_grids_txt0("atom_" + std::to_string(atom) + "_grid.txt", points);
        }
    }


    chemgrid::grid::PartitionOut part_h2o;
    // partition 部分
    auto start2 = std::chrono::high_resolution_clock::now(); 
    //原子坐标，grouped 
    part_h2o = chemgrid::grid::get_partition_from_templates_GPU_nbr2_1kernel_cut(atm_coords, grouped, charges, chemgrid::RadiiAdjustMethod::Treutler, chemgrid::constants::BRAGG_RADII_ANG, chemgrid::BeckeScheme::Original, true);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end2 - start2;
    std::cout << "partition: " << duration2.count() << " seconds" << std::endl;


    export_grids_txt("h2o_partition.txt", part_h2o.first, part_h2o.second);
    std::cout << "H2O partition grid has " << part_h2o.first.size() << " points\n";
    // 全局化
    std::vector<int> atm_idx;               // 输出
    std::vector<double> quadrature_weights; // 输出
    auto start3 = std::chrono::high_resolution_clock::now();
    chemgrid::grid::build_atm_idx_and_weights_by_Z(grouped, charges, part_h2o.first, part_h2o.second, atm_idx, quadrature_weights);
    // export_grid2_txt("h2o_global.txt", part_h2o.first, quadrature_weights, atm_idx);

    // 重排
    std::vector<size_t> idx = chemgrid::grid::arg_group_grids_cpu(part_h2o.first, atm_coords,
                                                                  chemgrid::constants::GROUP_BOX_SIZE,          // 1.2
                                                                  chemgrid::constants::GROUP_BOUNDARY_PENALTY); // 4.2
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.first);
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.second);
    chemgrid::grid::reorder_by_index_inplace(idx, atm_idx);
    chemgrid::grid::reorder_by_index_inplace(idx, quadrature_weights);
    // export_grid3_txt("h2o_reorder.txt", part_h2o.first, part_h2o.second, quadrature_weights, atm_idx);

    // 填充
    chemgrid::grid::pad_grids_cpp(part_h2o.first, part_h2o.second, atm_idx, quadrature_weights, chemgrid::constants::ALIGNMENT_UNIT);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration3 = end3 - start3;
    std::cout << "result: " << duration3.count() << " seconds" << std::endl;
    export_grid3_txt("h2o_padding_test.txt", part_h2o.first, part_h2o.second, quadrature_weights, atm_idx);

    std::cout << "totel time: " << duration1.count()+ duration2.count()+ duration3.count() << " seconds" << std::endl;
    return 0;
}