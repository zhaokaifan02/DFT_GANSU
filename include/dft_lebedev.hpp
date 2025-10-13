#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <map>
namespace chemgrid
{

  // SoA：CUDA 访存更友好
  struct AngGrid
  {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> w;

    std::size_t size() const noexcept { return w.size(); }
    void reserve(std::size_t n)
    {
      x.reserve(n);
      y.reserve(n);
      z.reserve(n);
      w.reserve(n);
    }
    void push(double xi, double yi, double zi, double wi)
    {
      x.push_back(xi);
      y.push_back(yi);
      z.push_back(zi);
      w.push_back(wi);
    }
  };
  inline std::map<int, int> LEBEDEV_ORDER = {
      {0, 1},
      {3, 6},
      {5, 14},
      {7, 26},
      {9, 38},
      {11, 50},
      {13, 74},
      {15, 86},
      {17, 110},
      {19, 146},
      {21, 170},
      {23, 194},
      {25, 230},
      {27, 266},
      {29, 302},
      {31, 350},
      {35, 434},
      {41, 590},
      {47, 770},
      {53, 974},
      {59, 1202},
      {65, 1454},
      {71, 1730},
      {77, 2030},
      {83, 2354},
      {89, 2702},
      {95, 3074},
      {101, 3470},
      {107, 3890},
      {113, 4334},
      {119, 4802},
      {125, 5294},
      {131, 5810}
    };
  // Lebedev 点数支持：返回可用点数列表（用于参数检查/修剪选择）
  const std::vector<int> &lebedev_supported_sizes();
  const std::vector<int> LEBEDEV_NGRID = {
      6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350,
      434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074,
      3470, 3890, 4334, 4802, 5294, 5810
    };
  // 生成指定点数的 Lebedev 角向网格（单位球面，权重与表保持一致）
  AngGrid make_lebedev_grid(int n_points);

  // —— 设备端包装（可选，不依赖 CUDA 头，避免在通用头里 include cuda_runtime）——
  struct DeviceAngGrid
  {
    double *dx{nullptr};
    double *dy{nullptr};
    double *dz{nullptr};
    double *dw{nullptr};
    std::size_t n{0};

    // 在 .cu 实现里提供这两个函数（或在调用处自己写）
    void upload_from(const AngGrid &h); // cudaMalloc + cudaMemcpy
    void free();                        // cudaFree
  };

} // namespace chemgrid
