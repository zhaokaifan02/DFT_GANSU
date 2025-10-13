#include "dft_constant.hpp"


double chemgrid::constants::bragg_radius_bohr(std::uint32_t Z)
{
    const std::size_t n = sizeof(BRAGG_RADII_ANG) / sizeof(double);
    if (Z < n)
    {
        const double ang = BRAGG_RADII_ANG[Z];
        // 允许 Ghost/unknown（非 NaN）也被正常换算
        return ANG2BOHR * ang;
    }
    return ANG2BOHR;
}

double chemgrid::constants::sg1_radius_bohr(std::uint32_t Z)
    {
        const std::size_t n = sizeof(SG1_RADII_BOHR) / sizeof(double);
        if (Z < n)
        {
            const double v = SG1_RADII_BOHR[Z];
            return v;
        }
        // 回退：SG1 未定义的元素使用 BRAGG
        return bragg_radius_bohr(Z);
    }