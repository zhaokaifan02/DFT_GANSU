
#include "dft_genGrid.hpp"
#include <stdio.h>
#include "dft_aoEVAL.hpp"
#include "dft_shell_eval.hpp"
std::vector<int> unique_Z(const std::vector<int> &Z)
{
    std::vector<int> uniq = Z;
    std::sort(uniq.begin(), uniq.end());                           // 排序
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end()); // 去重
    return uniq;
}

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
        throw std::invalid_argument("export_grid_txt: The input array sizes are inconsistent.");

    std::ofstream ofs(filename);
    if (!ofs)
        throw std::runtime_error("file open error: " + filename);

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
    std::getline(ifs, line);
    std::getline(ifs, line);

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

void chemgrid::gen_grid(std::string tag, std::string basis)
{
    const std::string final_out = "./grids/" + tag + "_grid.txt";
    const std::string final_outAO = "./grids/" + tag + "_AO.txt";
    const std::string input_file = "./xyz/" + tag + ".txt";
    // ======================================================
    std::vector<int> charges;
    std::vector<std::array<double, 3>> atm_coords;
    chemgrid::dft_getMolfromTXT(tag, charges, atm_coords);
    std::vector<int> u_z = chemgrid::gbs::unique_Z_c(charges);
    // grids generate
    std::pair<std::vector<std::array<double, 3>>, std::vector<double>> grids = chemgrid::dft_gen_grid(charges, atm_coords);
    std::vector<int> atm_idx(grids.first.size(), 0); // 占位符
    export_GRID_txt(final_out, grids.first, grids.second, grids.second, atm_idx);
    std::map<int, std::vector<atom_AO>> normed_bas = chemgrid::gbs::get_normed_ao("./" + basis + ".gbs", charges);
    std::vector<AODesc> AODESC = chemgrid::gbs::generate_ao_list(normed_bas, charges, atm_coords);
    int ngrids = static_cast<int>(grids.first.size());
    int nao = static_cast<int>(AODESC.size());
    double *ao_values = new double[ngrids * nao];
    chemgrid::AOEval::evaluate_aos_on_grids_gpu_raw(AODESC, atm_coords, grids.first, ao_values, ngrids, nao);
    write_ao_txt(final_outAO, ao_values, ngrids, nao);
    std::cout << "AO written to ao.txt (" << ngrids << " x " << nao << ")" << std::endl;
}

void chemgrid::gen_grid_screen(std::string tag, std::string basis)
{
    // todo screen select
    chemgrid::gen_grid(tag, basis);
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
        }
    }
    // becke
    chemgrid::grid::PartitionOut part_h2o;
    part_h2o = chemgrid::grid::get_partition_from_templates_GPU_warp_ownerOnly(
        atm_coords, grouped, charges,
        chemgrid::constants::BRAGG_RADII_ANG);

    // reorder
    std::vector<int> atm_idx;
    std::vector<double> quadrature_weights;
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
        }
    }
    // becke
    chemgrid::grid::PartitionOut part_h2o;
    part_h2o = chemgrid::grid::get_partition_from_templates_GPU_warp_ownerOnly(
        atm_coords, grouped, charges,
        chemgrid::constants::BRAGG_RADII_ANG);

    // reorder
    std::vector<int> atm_idx;
    std::vector<double> quadrature_weights;
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
    const std::string input_file = "./xyz/"+ tag + ".txt";
    // 分子data
    read_mol_txt(input_file, charges, atm_coords);
}

std::map<int, std::vector<atom_AO>> chemgrid::dft_get_normed_basis(gansu::BasisSet bs, std::vector<int> charges)
{

    std::vector<int> u_z = chemgrid::gbs::unique_Z_c(charges);
    // 存储基组
    std::vector<AODesc> aos;
    int atom_index = 0; // 原子索引
    std::map<int, std::vector<atom_AO>> normed_bas;
    // 读取基组
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
            int l = chemgrid::gbs::l_from_shell_char(type[0]);
            std::vector<gansu::PrimitiveGauss> primitives = a.get_primitives();
            std::vector<double> esp;
            std::vector<double> c;
            for (gansu::PrimitiveGauss b : primitives)
            {
                // std::cout << "esp: " << b.exponent << "  c: " << b.coefficient << "\n";
                esp.push_back(b.exponent);
                c.push_back(b.coefficient);
            }
            // 处理
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

chemgrid::AOGrids chemgrid::dft_gen_ao(gansu::BasisSet &bs, std::vector<int> charges, std::vector<std::array<double, 3>> &atm_coords, std::vector<std::array<double, 3>> coords)
{
    std::map<int, std::vector<atom_AO>> normed_bas = chemgrid::dft_get_normed_basis(bs, charges);
    std::vector<AODesc> AODESC = chemgrid::gbs::generate_ao_list(normed_bas, charges, atm_coords);
    int ngrids = static_cast<int>(coords.size());
    int nao = static_cast<int>(AODESC.size());
    double *ao_values = new double[ngrids * nao];
    chemgrid::AOEval::evaluate_aos_on_grids_gpu_raw(AODESC, atm_coords, coords, ao_values, ngrids, nao);
    AOGrids outAO;
    outAO.ao=ao_values;
    outAO.naos = nao;
    outAO.ngrids = ngrids;
    return outAO;
}