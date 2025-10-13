// io_txt.cpp
#include "io_txt.hpp"
#include <fstream>
#include <stdexcept>

Eigen::MatrixXd load_matrix_txt(const std::string& path){
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("cannot open " + path);
    int n,m; ifs >> n >> m;
    Eigen::MatrixXd M(n,m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            ifs >> M(i,j);
    return M;
}

Eigen::VectorXd load_vector_txt(const std::string& path){
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("cannot open " + path);
    int n; ifs >> n;
    Eigen::VectorXd v(n);
    for(int i=0;i<n;i++) ifs >> v(i);
    return v;
}

double load_scalar_txt(const std::string& path){
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("cannot open " + path);
    double x; ifs >> x; return x;
}

std::vector<ERIEntry> load_eri_entries(const std::string& path, int &nao_out){
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("cannot open " + path);
    int nao; ifs >> nao;
    nao_out = nao;
    std::vector<ERIEntry> entries;
    int i,j,k,l; double v;
    while(ifs >> i >> j >> k >> l >> v){
        entries.push_back({i,j,k,l,v});
    }
    return entries;
}

void load_AO_txt(const std::string& path, int &ngrid, int &nao, std::vector<double> &ao_rowmajor){
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("cannot open " + path);
    ifs >> ngrid >> nao;
    ao_rowmajor.resize(size_t(ngrid) * size_t(nao));
    for(int g=0; g<ngrid; ++g){
        for(int a=0; a<nao; ++a){
            ifs >> ao_rowmajor[size_t(g)*size_t(nao) + a];
        }
    }
}
