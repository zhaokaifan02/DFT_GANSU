#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include "shell_eval.hpp"
namespace chemgrid::gbs
{

    // ============ 基本数据结构 ============

    // 一条“壳”（已把 SP 拆分为独立 S 或 P）
    struct Shell
    {
        char type;                  // 'S','P','D','F','G'...
        int l;                      // 0,1,2,3,...
        std::vector<double> exps;   // α_j
        std::vector<double> coeffs; // c_j (原始系数，尚未含 primitive/分量归一化)
    };

    // 元素基组：同一元素下的多个壳
    struct ElementBasis
    {
        std::vector<Shell> shells;
    };

    // ============ 实用函数：角动量/归一化 ============

    inline int l_from_shell_char(char c)
    {
        switch (c)
        {
        case 'S':
            return 0;
        case 'P':
            return 1;
        case 'D':
            return 2;
        case 'F':
            return 3;
        case 'G':
            return 4;
        case 'H':
            return 5;
        default:
            throw std::runtime_error(std::string("Unsupported shell type: ") + c);
        }
    }

    // 双阶乘 (odd) : (2n-1)!! ；当传入负数或 0，返回 1
    inline double double_factorial_odd(int n)
    {
        if (n <= 0)
            return 1.0;
        double v = 1.0;
        for (int k = n; k >= 1; k -= 2)
            v *= k;
        return v;
    }

    // 笛卡尔 primitive 归一化常数：
    // N = [ (2α/π)^(3/2) * (4α)^(lx+ly+lz) / ((2lx-1)!!(2ly-1)!!(2lz-1)!!) ]^(1/2)
    inline double prim_norm_cart(double alpha, int lx, int ly, int lz)
    {
        const int L = lx + ly + lz;
        const double two_alpha = 2.0 * alpha;
        const double pref = pow(two_alpha / M_PI, 1.5);
        const double num = pow(4.0 * alpha, L);
        const double den = double_factorial_odd(2 * lx - 1) * double_factorial_odd(2 * ly - 1) * double_factorial_odd(2 * lz - 1);
        return sqrt(pref * num / den);
    }

    // ============ 主类：GBSReader ============

    class GBSReader
    {
    public:
        // 读取并解析 .gbs 文件；支持 "****" 分隔多个元素；支持 "D" 指数（Fortran 风格）
        void build(const std::string &path);

        bool has_element(const std::string &symbol) const;
        const ElementBasis &element_basis(const std::string &symbol) const;

        // 列出所有已加载的元素符号（用于调试/检查）
        std::vector<std::string> elements() const;

    private:
        std::unordered_map<std::string, ElementBasis> basis_map_;

        // 解析工具
        static double parse_number(const std::string &tok); // 支持 D 指数
        static bool is_element_header(const std::string &line, std::string &symbol);
        static bool is_shell_header(const std::string &line, char &type, int &nprim, double &scale);

        // 内部添加：若遇到 "SP" 则拆成 S/P 两个壳
        void add_shell_block(const std::string &type_str,
                             int nprim, double scale,
                             const std::vector<std::array<double, 3>> &rows_SP,     // 对于 SP：{alpha, cs, cp}
                             const std::vector<std::pair<double, double>> &rows_1C, // 对于单列系数：{alpha, c}
                             ElementBasis &ebs);
    };
    std::vector<AODesc>
    build_aodescs_from_gbs(
        chemgrid::gbs::GBSReader &reader,
        const std::vector<int> &charges,                      // 原子序数 Z
        const std::vector<std::array<double, 3>> &atm_coords// A×3 (Bohr)
        );

} // namespace gbs
