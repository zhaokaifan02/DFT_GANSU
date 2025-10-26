#include "dft_gbsReader.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdlib> // strtod
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>

namespace chemgrid::gbs
{

    // 你自己已有的类型：Shell / ElementBasis / l_from_shell_char / GBSReader 等
    // struct Shell { char type; int l; std::vector<double> exps, coeffs; };
    // struct ElementBasis { std::vector<Shell> shells; };
    // int l_from_shell_char(char); // S->0, P->1, D->2, ...
    // class GBSReader { ... private: std::unordered_map<std::string, ElementBasis> basis_map_; ... };

    static inline std::string trim(const std::string &s)
    {
        size_t b = 0, e = s.size();
        while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
            ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
            --e;
        return s.substr(b, e - b);
    }

    static inline std::vector<std::string> split_ws(const std::string &s)
    {
        std::istringstream iss(s);
        std::vector<std::string> toks;
        std::string t;
        while (iss >> t)
            toks.push_back(t);
        return toks;
    }

    /* ---------- 元素识别（严格表） ---------- */
    static inline const std::unordered_set<std::string> &element_set()
    {
        static const std::unordered_set<std::string> S = {
            "H", "He",
            "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
            "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe",
            "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn",
            "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
            "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
        return S;
    }

    static inline std::string canon_elem(std::string s)
    {
        for (auto &c : s)
            c = (char)std::tolower((unsigned char)c);
        if (!s.empty())
            s[0] = (char)std::toupper((unsigned char)s[0]);
        return s;
    }

    /* ---------- 数字解析（宽容） ---------- */
    double GBSReader::parse_number(const std::string &tok)
    {
        std::string t = trim(tok);
        if (t.empty())
            throw std::invalid_argument("parse_number: empty token");
        for (auto &ch : t)
            if (ch == 'D' || ch == 'd')
                ch = 'E';
        t.erase(std::remove(t.begin(), t.end(), ','), t.end());
        auto ok = [](char c)
        {
            return std::isdigit((unsigned char)c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E';
        };
        size_t len = 0;
        while (len < t.size() && ok(t[len]))
            ++len;
        if (len == 0)
            throw std::invalid_argument("parse_number: not a number token=[" + tok + "]");
        std::string pre = t.substr(0, len);
        char *endp = nullptr;
        errno = 0;
        double v = std::strtod(pre.c_str(), &endp);
        if (endp == pre.c_str() || errno == ERANGE)
            throw std::invalid_argument("parse_number: cannot parse token=[" + tok + "]");
        return v;
    }

    /* ---------- 元素/壳头识别 ---------- */
    // 行: "O 0" / "He" / "C    0"
    bool GBSReader::is_element_header(const std::string &line, std::string &symbol)
    {
        auto s = trim(line);
        if (s.empty() || s == "****")
            return false;
        if (s[0] == '!' || s[0] == '#')
            return false;
        auto toks = split_ws(s);
        if (toks.empty())
            return false;

        std::string cand = canon_elem(toks[0]);
        if (element_set().count(cand) == 0)
            return false;

        if (toks.size() >= 2)
        {
            const std::string &t1 = toks[1];
            bool all_int = !t1.empty();
            for (char c : t1)
            {
                if (!std::isdigit((unsigned char)c) && c != '+' && c != '-')
                {
                    all_int = false;
                    break;
                }
            }
            if (!all_int)
                return false; // 避免把 "SP 3 1.00" 误判为元素
        }
        symbol = cand;
        return true;
    }

    static inline bool is_shell_type(std::string t)
    {
        if (t == "L")
            t = "SP"; // 兼容 L==SP
        if (t == "SP")
            return true;
        if (t.size() != 1)
            return false;
        char c = (char)std::toupper((unsigned char)t[0]);
        return c == 'S' || c == 'P' || c == 'D' || c == 'F' || c == 'G' || c == 'H' || c == 'I';
    }

    // 行: "S 3 1.00" / "SP 3 1.00" / "D 3"
    bool GBSReader::is_shell_header(const std::string &line, char &type, int &nprim, double &scale)
    {
        auto s = trim(line);
        if (s.empty())
            return false;
        if (s[0] == '!' || s[0] == '#')
            return false;
        auto toks = split_ws(s);
        if (toks.size() < 2)
            return false;

        std::string T = toks[0];
        if (!is_shell_type(T))
            return false;

        try
        {
            nprim = std::stoi(toks[1]);
        }
        catch (...)
        {
            return false;
        }

        scale = 1.0;
        if (toks.size() >= 3)
        {
            try
            {
                scale = parse_number(toks[2]);
            }
            catch (...)
            {
                scale = 1.0;
            }
        }
        type = (T == "SP" ? 'S' : (char)std::toupper((unsigned char)T[0]));
        return true;
    }

    /* ---------- 壳写入 ---------- */
    void GBSReader::add_shell_block(const std::string &type_str,
                                    int nprim, double scale,
                                    const std::vector<std::array<double, 3>> &rows_SP,
                                    const std::vector<std::pair<double, double>> &rows_1C,
                                    ElementBasis &ebs)
    {
        if (type_str == "SP" || type_str == "L")
        {
            Shell shS, shP;
            shS.type = 'S';
            shS.l = l_from_shell_char('S');
            shP.type = 'P';
            shP.l = l_from_shell_char('P');
            shS.exps.reserve(nprim);
            shS.coeffs.reserve(nprim);
            shP.exps.reserve(nprim);
            shP.coeffs.reserve(nprim);
            for (int i = 0; i < nprim; ++i)
            {
                const double alpha = rows_SP[i][0];
                const double cs = rows_SP[i][1] * scale;
                const double cp = rows_SP[i][2] * scale;
                shS.exps.push_back(alpha);
                shS.coeffs.push_back(cs);
                shP.exps.push_back(alpha);
                shP.coeffs.push_back(cp);
            }
            ebs.shells.push_back(std::move(shS));
            ebs.shells.push_back(std::move(shP));
            return;
        }

        if (rows_1C.size() != static_cast<size_t>(nprim))
            throw std::runtime_error("GBSReader internal: rows_1C size mismatch");

        Shell sh;
        sh.type = type_str[0];
        sh.l = l_from_shell_char(sh.type);
        sh.exps.reserve(nprim);
        sh.coeffs.reserve(nprim);
        for (int i = 0; i < nprim; ++i)
        {
            sh.exps.push_back(rows_1C[i].first);
            sh.coeffs.push_back(rows_1C[i].second * scale);
        }
        ebs.shells.push_back(std::move(sh));
    }

    /* ---------- 主读取 ---------- */
    void GBSReader::build(const std::string &path)
    {
        basis_map_.clear();

        std::ifstream fin(path);
        if (!fin)
            throw std::runtime_error("Cannot open gbs file: " + path);

        std::string line, cur_elem;
        ElementBasis *cur_ebs = nullptr;

        enum class State
        {
            NeedElem,
            NeedShellHeader,
            InShellRows
        };
        State st = State::NeedElem;

        std::string cur_type_str;
        int cur_nprim = 0;
        double cur_scale = 1.0;
        bool is_SP = false;
        std::vector<std::array<double, 3>> rows_SP;
        std::vector<std::pair<double, double>> rows_1C;

        auto flush_shell_block = [&]()
        {
            if (cur_nprim <= 0)
                return;
            if (!cur_ebs)
                throw std::runtime_error("Shell without element.");
            add_shell_block(cur_type_str, cur_nprim, cur_scale, rows_SP, rows_1C, *cur_ebs);
            cur_type_str.clear();
            cur_nprim = 0;
            cur_scale = 1.0;
            is_SP = false;
            rows_SP.clear();
            rows_1C.clear();
        };

        auto safe_parse = [&](const std::string &tok, const std::string &whole) -> double
        {
            try
            {
                return parse_number(tok);
            }
            catch (const std::exception &e)
            {
                std::ostringstream oss;
                oss << "parse_number failed: token=[" << tok << "], line=[" << whole << "], reason=" << e.what();
                throw std::invalid_argument(oss.str());
            }
        };
        auto can_num = [&](const std::string &tok) -> bool
        {
            try
            {
                (void)parse_number(tok);
                return true;
            }
            catch (...)
            {
                return false;
            }
        };

        while (std::getline(fin, line))
        {
            std::string s = trim(line);
            if (s.empty())
                continue;
            if (s[0] == '!' || s[0] == '#')
                continue;

            if (s == "****")
            {
                flush_shell_block();
                st = State::NeedElem;
                cur_elem.clear();
                cur_ebs = nullptr;
                continue;
            }

            if (st == State::NeedElem)
            {
                std::string sym;
                if (is_element_header(s, sym))
                {
                    cur_elem = sym;
                    cur_ebs = &basis_map_[cur_elem];
                    st = State::NeedShellHeader;
                }
                continue;
            }

            if (st == State::NeedShellHeader)
            {
                char t;
                int n;
                double sc;
                if (is_shell_header(s, t, n, sc))
                {
                    flush_shell_block();
                    cur_nprim = n;
                    cur_scale = sc;
                    cur_type_str = split_ws(s)[0];
                    if (cur_type_str == "L")
                        cur_type_str = "SP";
                    is_SP = (cur_type_str == "SP");
                    rows_SP.clear();
                    rows_1C.clear();
                    st = State::InShellRows;
                }
                continue;
            }

            if (st == State::InShellRows)
            {
                auto toks = split_ws(s);
                const bool full_SP = is_SP && (int)rows_SP.size() >= cur_nprim;
                const bool full_1C = !is_SP && (int)rows_1C.size() >= cur_nprim;

                const bool looks_data_SP = is_SP && !full_SP && toks.size() >= 3 && can_num(toks[0]);
                const bool looks_data_1C = !is_SP && !full_1C && toks.size() >= 2 && can_num(toks[0]);

                if (looks_data_SP || looks_data_1C)
                {
                    if (is_SP)
                    {
                        std::array<double, 3> row;
                        row[0] = safe_parse(toks[0], s);
                        row[1] = safe_parse(toks[1], s);
                        row[2] = safe_parse(toks[2], s);
                        rows_SP.push_back(row);
                    }
                    else
                    {
                        double a = safe_parse(toks[0], s);
                        double c = safe_parse(toks[1], s);
                        rows_1C.emplace_back(a, c);
                    }
                    continue;
                }
                else
                {
                    // 新段落开始：先刷入当前壳
                    flush_shell_block();

                    std::string sym;
                    char t2;
                    int n2;
                    double sc2;
                    if (is_element_header(s, sym))
                    {
                        cur_elem = sym;
                        cur_ebs = &basis_map_[cur_elem];
                        st = State::NeedShellHeader;
                        continue;
                    }
                    else if (is_shell_header(s, t2, n2, sc2))
                    {
                        cur_nprim = n2;
                        cur_scale = sc2;
                        std::string tt = split_ws(s)[0];
                        cur_type_str = (tt == "L" ? "SP" : tt);
                        is_SP = (cur_type_str == "SP");
                        rows_SP.clear();
                        rows_1C.clear();
                        st = State::InShellRows;
                        continue;
                    }
                    else if (s == "****")
                    {
                        st = State::NeedElem;
                        cur_elem.clear();
                        cur_ebs = nullptr;
                        continue;
                    }
                    else
                    {
                        continue; // 注释或无关
                    }
                }
            }
        }

        flush_shell_block();
    }

    /* ---------- 查询辅助 ---------- */
    bool GBSReader::has_element(const std::string &symbol) const
    {
        return basis_map_.find(symbol) != basis_map_.end();
    }

    const ElementBasis &GBSReader::element_basis(const std::string &symbol) const
    {
        auto it = basis_map_.find(symbol);
        if (it == basis_map_.end())
            throw std::runtime_error("Element not found in gbs: " + symbol);
        return it->second;
    }

    std::vector<std::string> GBSReader::elements() const
    {
        std::vector<std::string> keys;
        keys.reserve(basis_map_.size());
        for (auto &kv : basis_map_)
            keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        return keys;
    }
    // ================= 实用：Z → 元素符号 =================
    static std::string Z_to_symbol(int Z)
    {
        static const char *tbl[] = {
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr"};
        if (Z >= 1 && Z <= (int)(sizeof(tbl) / sizeof(tbl[0])))
            return tbl[Z - 1];
        throw std::runtime_error("Z_to_symbol: unsupported Z=" + std::to_string(Z));
    }
    std::vector<AODesc>
    build_aodescs_from_gbs(
        chemgrid::gbs::GBSReader &reader,
        const std::vector<int> &charges,                     // 原子序数 Z
        const std::vector<std::array<double, 3>> &atm_coords // A×3 (Bohr)
    )
    {
        if (charges.size() != atm_coords.size())
            throw std::runtime_error("build_aodescs_from_gbs: charges and atm_coords size mismatch");

        std::vector<AODesc> out;
        out.reserve(128);

        for (int A = 0; A < (int)charges.size(); ++A)
        {
            const std::string sym = Z_to_symbol(charges[A]);

            // 从 reader 拿这个元素的所有壳
            const auto &ebs = reader.element_basis(sym); // 你的 GBSReader 提供的接口

            for (const auto &sh : ebs.shells)
            {
                const char t = sh.type;             // 'S' / 'P' （你说最多到 SP）
                const int l = l_from_shell_char(t); // 0 or 1
                const auto &exps = sh.exps;
                const auto &coeffs = sh.coeffs;

                if (exps.size() != coeffs.size())
                    throw std::runtime_error("gbs shell exps/coeffs size mismatch for " + sym);

                // 枚举笛卡尔分量
                std::vector<std::tuple<int, int, int>> triples;
                cartesian_triples_for_l(l, triples);

                for (auto [lx, ly, lz] : triples)
                {
                    // 复制原始系数，并按选项做归一化
                    std::vector<double> cj(coeffs);

                    for (size_t j = 0; j < cj.size(); ++j)
                    {
                        cj[j] *= prim_norm_cart(exps[j], lx, ly, lz);
                    }

                    AODesc ao;
                    ao.atom = A;
                    ao.l = l;
                    ao.lx = lx;
                    ao.ly = ly;
                    ao.lz = lz;
                    ao.exps = exps; // 共享 α
                    ao.coeffs = std::move(cj);
                    out.emplace_back(std::move(ao));
                }
            }
        }

        return out;
    }
} // namespace chemgrid::gbs