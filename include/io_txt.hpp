// io_txt.hpp
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

struct ERIEntry { int i,j,k,l; double v; };

Eigen::MatrixXd load_matrix_txt(const std::string& path);     // 首行: n m
Eigen::VectorXd load_vector_txt(const std::string& path);     // 首行: n
double          load_scalar_txt(const std::string& path);     // 单行一个数
std::vector<ERIEntry> load_eri_entries(const std::string& path, int &nao_out);

void load_AO_txt(const std::string& path, int &ngrid, int &nao, std::vector<double> &ao_rowmajor);
