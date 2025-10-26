
#include "dft_genGrid.hpp"
#include <stdio.h>
std::vector<int> unique_Z(const std::vector<int> &Z)
{
    std::vector<int> uniq = Z;
    std::sort(uniq.begin(), uniq.end());                           // 排序
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
void export_grid4_txt(const std::string &filename,
                      const std::vector<std::array<double, 3>> &coords,
                      const std::vector<double> &weights,
                      const std::vector<double> &quadrature_weights,
                      const std::vector<int> &atm_idx)
{
    const std::size_t N = coords.size();
    // std::cout << "N=" << N << ", weights.size()=" << weights.size()
    //           << ", quadrature_weights.size()=" << quadrature_weights.size()
    //           << ", atm_idx.size()=" << atm_idx.size() << std::endl;

    // if (weights.size() != N || atm_idx.size() != N)
    //     throw std::invalid_argument("export_grid_txt: 输入数组大小不一致");

    std::ofstream ofs(filename);
    if (!ofs)
        throw std::runtime_error("无法打开输出文件: " + filename);

    ofs << std::scientific << std::setprecision(20);
    for (std::size_t i = 0; i < N; ++i)
    {
        ofs << 0 << " "
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
static inline bool mul_overflow(size_t a, size_t b, size_t *out)
{
    if (a == 0 || b == 0)
    {
        *out = 0;
        return false;
    }
    if (a > std::numeric_limits<size_t>::max() / b)
        return true;
    *out = a * b;
    return false;
}

void write_ao_ngrid_nao_txt(const std::string &path,
                            const double *AO_naoG,
                            size_t ngrid, size_t nao)
{
    if (!AO_naoG)
        throw std::runtime_error("AO_naoG is null");
    size_t total = 0;
    if (mul_overflow(ngrid, nao, &total))
    {
        throw std::runtime_error("ngrid*nao overflows size_t");
    }

    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("Cannot open file: " + path);

    ofs << ngrid << " " << nao << "\n";
    ofs << std::setprecision(17) << std::scientific;

    for (size_t g = 0; g < ngrid; ++g)
    {
        for (size_t a = 0; a < nao; ++a)
        {
            size_t idx = a * ngrid + g; // 列主序
            if (idx >= total)
            { // 防止传参写反
                throw std::runtime_error("Index OOB: a=" + std::to_string(a) +
                                         " g=" + std::to_string(g) +
                                         " idx=" + std::to_string(idx) +
                                         " total=" + std::to_string(total) +
                                         " (did you swap ngrid/nao?)");
            }
            ofs << AO_naoG[idx];
            if (a + 1 < nao)
                ofs << ' ';
        }
        ofs << '\n';
    }
}

void chemgrid::gen_grid(std::string tag)
{
    const std::string input_file = tag + ".txt";
    const std::string final_out = tag + "_grid.txt";
    // ======================================================

    std::vector<double> x, y, z, w;
    std::vector<int> atom_id;

    // 分子data
    std::vector<int> charges;
    std::vector<std::array<double, 3>> atm_coords;
    read_mol_txt(input_file, charges, atm_coords);

    std::vector<int> atom_nuc = unique_Z(charges);

    // 原子网格生成
    chemgrid::grid::gen_atom_grid(x, y, z, w, atom_id, atom_nuc,
                                  chemgrid::RadialMethod::TreutlerAhlrichs, 3,
                                  chemgrid::PruningMethod::NWChem);
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
            // 可选：把每个原子的原子网格也带上 tag 前缀（不要求，但更清晰）
            // export_grids_txt0(tag + "_atom_" + std::to_string(atom) + "_grid.txt", points);
        }
    }
    // 分区becke
    chemgrid::grid::PartitionOut part_h2o;
    part_h2o = chemgrid::grid::get_partition_from_templates_GPU_warp_ownerOnly(
        atm_coords, grouped, charges,
        chemgrid::constants::BRAGG_RADII_ANG);

    // 输出准备
    std::vector<int> atm_idx;               // 输出
    std::vector<double> quadrature_weights; // 输出
    chemgrid::grid::build_atm_idx_and_weights_by_Z(grouped, charges, part_h2o.first, part_h2o.second, atm_idx, quadrature_weights);
    std::vector<size_t> idx = chemgrid::grid::arg_group_grids_cpu(part_h2o.first, atm_coords,
                                                                  chemgrid::constants::GROUP_BOX_SIZE,          // 1.2
                                                                  chemgrid::constants::GROUP_BOUNDARY_PENALTY); // 4.2
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.first);
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.second);
    chemgrid::grid::reorder_by_index_inplace(idx, atm_idx);
    chemgrid::grid::reorder_by_index_inplace(idx, quadrature_weights);
    chemgrid::grid::pad_grids_cpp(part_h2o.first, part_h2o.second, atm_idx, quadrature_weights, chemgrid::constants::ALIGNMENT_UNIT);
    export_grid3_txt(final_out, part_h2o.first, part_h2o.second, quadrature_weights, atm_idx);
    gansu::BasisSet bs_;
    std::ifstream fin("./basis/sto-3g.gbs");
    if (!fin)
    {
        throw std::runtime_error("Cannot open gbs: ");
    }

    gansu::BasisSet bs = bs_.construct_from_gbs("./basis/sto-3g.gbs");
    auto aos = build_ao_descs_from_mol_and_gbs(charges, atm_coords, bs, {/*normalize_cart*/ true});
    int nao = (int)aos.size();
    int G = (int)part_h2o.first.size();
    std::vector<double> AO; // 大小 = nao * G

    evaluate_AO_on_grid(aos, atm_coords, part_h2o.first, AO);
    write_ao_ngrid_nao_txt(tag + "_AO.txt", AO.data(), G, nao);
}

void chemgrid::gen_grid_screen(std::string tag)
{
    const std::string input_file = tag + ".txt";
    const std::string final_out = tag + "_grid.txt";
    // ======================================================

    std::vector<double> x, y, z, w;
    std::vector<int> atom_id;

    // 分子data
    std::vector<int> charges;
    std::vector<std::array<double, 3>> atm_coords;
    read_mol_txt(input_file, charges, atm_coords);

    std::vector<int> atom_nuc = unique_Z(charges);

    // 原子网格生成
    chemgrid::grid::gen_atom_grid(x, y, z, w, atom_id, atom_nuc,
                                  chemgrid::RadialMethod::TreutlerAhlrichs, 3,
                                  chemgrid::PruningMethod::NWChem);
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
            // 可选：把每个原子的原子网格也带上 tag 前缀（不要求，但更清晰）
            // export_grids_txt0(tag + "_atom_" + std::to_string(atom) + "_grid.txt", points);
        }
    }
    // 分区becke
    chemgrid::grid::PartitionOut part_h2o;
    part_h2o = chemgrid::grid::get_partition_from_templates_GPU_warp_ownerOnly(
        atm_coords, grouped, charges,
        chemgrid::constants::BRAGG_RADII_ANG);

    // 输出准备
    std::vector<int> atm_idx;               // 输出
    std::vector<double> quadrature_weights; // 输出
    chemgrid::grid::build_atm_idx_and_weights_by_Z(grouped, charges, part_h2o.first, part_h2o.second, atm_idx, quadrature_weights);
    std::vector<size_t> idx = chemgrid::grid::arg_group_grids_cpu(part_h2o.first, atm_coords,
                                                                  chemgrid::constants::GROUP_BOX_SIZE,          // 1.2
                                                                  chemgrid::constants::GROUP_BOUNDARY_PENALTY); // 4.2
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.first);
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.second);
    chemgrid::grid::reorder_by_index_inplace(idx, atm_idx);
    chemgrid::grid::reorder_by_index_inplace(idx, quadrature_weights);
    chemgrid::grid::pad_grids_cpp(part_h2o.first, part_h2o.second, atm_idx, quadrature_weights, chemgrid::constants::ALIGNMENT_UNIT);
    export_grid3_txt(final_out, part_h2o.first, part_h2o.second, quadrature_weights, atm_idx);
    gansu::BasisSet bs_;
    std::ifstream fin("./basis/sto-3g.gbs");
    if (!fin)
    {
        throw std::runtime_error("Cannot open gbs: ");
    }

    gansu::BasisSet bs = bs_.construct_from_gbs("./basis/sto-3g.gbs");
    auto aos = build_ao_descs_from_mol_and_gbs(charges, atm_coords, bs, {/*normalize_cart*/ true});
    int nao = (int)aos.size();
    int G = (int)part_h2o.first.size();
    std::vector<double> AO; // 大小 = nao * G
    std::vector<uint8_t> screen_mask = build_screen_index_no_omp(
        aos, atm_coords, part_h2o.first,
        56, // blksize
        100 // nbins
    );
    long long sum_screen = 0;
    for (int i = 0; i < screen_mask.size(); i++)
    {
        if (screen_mask[i] == 0)
            sum_screen++;
        // printf("1");
    }
    std::cout << "Screened points: " << sum_screen << " / " << screen_mask.size() << std::endl;
    printf("Screen mask built, nao=%d, nblk=%d\n", nao, (G + 56 - 1) / 56);
    evaluate_AO_on_grid_screened(aos, atm_coords, part_h2o.first, 56, screen_mask, AO);
    write_ao_ngrid_nao_txt(tag + "_SCREEN_AO.txt", AO.data(), G, nao);
}
std::pair<std::vector<std::array<double, 3>>, std::vector<double>> chemgrid::dft_gen_grid(std::vector<int> charges, std::vector<std::array<double, 3>> atm_coords)
{
    std::vector<int> atom_nuc = unique_Z(charges);
    std::vector<double> x, y, z, w;
    std::vector<int> atom_id;
    // 原子网格生成

    chemgrid::grid::gen_atom_grid(x, y, z, w, atom_id, atom_nuc,
                                  chemgrid::RadialMethod::TreutlerAhlrichs, 3,
                                  chemgrid::PruningMethod::NWChem);
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
            // 可选：把每个原子的原子网格也带上 tag 前缀（不要求，但更清晰）
            // export_grids_txt0(tag + "_atom_" + std::to_string(atom) + "_grid.txt", points);
        }
    }
    // 分区becke
    chemgrid::grid::PartitionOut part_h2o;
    part_h2o = chemgrid::grid::get_partition_from_templates_GPU_warp_ownerOnly(
        atm_coords, grouped, charges,
        chemgrid::constants::BRAGG_RADII_ANG);

    // 输出准备
    std::vector<int> atm_idx;               // 输出
    std::vector<double> quadrature_weights; // 输出
    chemgrid::grid::build_atm_idx_and_weights_by_Z(grouped, charges, part_h2o.first, part_h2o.second, atm_idx, quadrature_weights);
    std::vector<size_t> idx = chemgrid::grid::arg_group_grids_cpu(part_h2o.first, atm_coords,
                                                                  chemgrid::constants::GROUP_BOX_SIZE,          // 1.2
                                                                  chemgrid::constants::GROUP_BOUNDARY_PENALTY); // 4.2
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.first);
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.second);
    chemgrid::grid::reorder_by_index_inplace(idx, atm_idx);
    chemgrid::grid::reorder_by_index_inplace(idx, quadrature_weights);
    chemgrid::grid::pad_grids_cpp(part_h2o.first, part_h2o.second, atm_idx, quadrature_weights, chemgrid::constants::ALIGNMENT_UNIT);
    export_grid3_txt("GRIDS.txt", part_h2o.first, part_h2o.second, quadrature_weights, atm_idx);
    return {part_h2o.first, part_h2o.second};
}
std::pair<std::vector<std::array<double, 3>>, std::vector<double>> chemgrid::dft_gen_grid(std::string tag)
{
    const std::string input_file = tag + ".txt";
    const std::string final_out = tag + "_grid.txt";
    // ======================================================

    std::vector<double> x, y, z, w;
    std::vector<int> atom_id;

    // 分子data
    std::vector<int> charges;
    std::vector<std::array<double, 3>> atm_coords;
    read_mol_txt(input_file, charges, atm_coords);

    std::vector<int> atom_nuc = unique_Z(charges);

    // 原子网格生成
    chemgrid::grid::gen_atom_grid(x, y, z, w, atom_id, atom_nuc,
                                  chemgrid::RadialMethod::TreutlerAhlrichs, 3,
                                  chemgrid::PruningMethod::NWChem);
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
            // 可选：把每个原子的原子网格也带上 tag 前缀（不要求，但更清晰）
            // export_grids_txt0(tag + "_atom_" + std::to_string(atom) + "_grid.txt", points);
        }
    }
    // 分区becke
    chemgrid::grid::PartitionOut part_h2o;
    part_h2o = chemgrid::grid::get_partition_from_templates_GPU_warp_ownerOnly(
        atm_coords, grouped, charges,
        chemgrid::constants::BRAGG_RADII_ANG);

    // 输出准备
    std::vector<int> atm_idx;               // 输出
    std::vector<double> quadrature_weights; // 输出
    chemgrid::grid::build_atm_idx_and_weights_by_Z(grouped, charges, part_h2o.first, part_h2o.second, atm_idx, quadrature_weights);
    std::vector<size_t> idx = chemgrid::grid::arg_group_grids_cpu(part_h2o.first, atm_coords,
                                                                  chemgrid::constants::GROUP_BOX_SIZE,          // 1.2
                                                                  chemgrid::constants::GROUP_BOUNDARY_PENALTY); // 4.2
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.first);
    chemgrid::grid::reorder_by_index_inplace(idx, part_h2o.second);
    chemgrid::grid::reorder_by_index_inplace(idx, atm_idx);
    chemgrid::grid::reorder_by_index_inplace(idx, quadrature_weights);
    chemgrid::grid::pad_grids_cpp(part_h2o.first, part_h2o.second, atm_idx, quadrature_weights, chemgrid::constants::ALIGNMENT_UNIT);
    return {part_h2o.first, part_h2o.second};
}

void chemgrid::dft_getMolfromTXT(std::string tag, std::vector<int> &charges, std::vector<std::array<double, 3>> &atm_coords)
{
    const std::string input_file = tag + ".txt";
    const std::string final_out = tag + "_grid.txt";
    // 分子data

    read_mol_txt(input_file, charges, atm_coords);
}