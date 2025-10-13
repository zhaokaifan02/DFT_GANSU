// rks_scf.cpp
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring> // memset
#include <iomanip>
#include "io_txt.hpp"

// ==== 你已有的实现（这里用声明，链接时把你的实现文件一并编译进来）====
void get_rho(int nao, int ngrid,
             const double *dm,
             const double *ao,
             double *rho_out)
{
    std::memset(rho_out, 0, sizeof(double) * ngrid);

    for (int g = 0; g < ngrid; ++g)
    {
        const double *phi_g = ao + g * nao;
        for (int u = 0; u < nao; ++u)
        {
            for (int v = 0; v < nao; ++v)
            {
                rho_out[g] += dm[u * nao + v] * phi_g[u] * phi_g[v];
            }
        }
    }
}

void build_coulomb_matrix(int nao,
                          const double *eri,
                          const double *dm,
                          double *J)
{
    std::memset(J, 0, sizeof(double) * nao * nao);
    for (int m = 0; m < nao; ++m)
        for (int n = 0; n < nao; ++n)
            for (int l = 0; l < nao; ++l)
                for (int s = 0; s < nao; ++s)
                    J[m * nao + n] += dm[l * nao + s] * eri[((m * nao + n) * nao + l) * nao + s];
}

struct VWNPar
{
    double A, b, c, x0;
};
static const VWNPar vwn_param[2] = {
    {0.0310907, 3.72744, 12.9352, -0.10498},
    {0.01554535, 7.06042, 18.0578, -0.32500}};

inline void lda_exc_vxc_impl(int n, const double *rho,
                             double *exc, double *vxc, double zeta)
{
    const double pi = 3.14159265358979323846;
    const double Cx = 0.7385587663820224;

    const double z2 = zeta * zeta;

    for (int i = 0; i < n; ++i)
    {
        double r = std::max(rho[i], 1e-300);
        double rs = std::pow(3.0 / (4.0 * pi * r), 1.0 / 3.0);
        double x = std::sqrt(rs);

        double ec0, dec0_dx, ec1, dec1_dx;
        auto vwn_ec = [](double x, const VWNPar &p, double &ec, double &dec_dx)
        {
            const double X = x * x + p.b * x + p.c;
            const double Q = std::sqrt(4.0 * p.c - p.b * p.b);
            const double log_term = std::log(x * x / X);
            const double atan_term = 2.0 * p.b / Q * std::atan(Q / (2.0 * x + p.b));
            const double x02 = p.x0 * p.x0;
            const double denom = x02 + p.b * p.x0 + p.c;
            const double corr = p.b * p.x0 / denom *
                                (std::log((x - p.x0) * (x - p.x0) / X) +
                                 2.0 * (2.0 * p.x0 + p.b) / Q * std::atan(Q / (2.0 * x + p.b)));
            ec = p.A * (log_term + atan_term - corr);
            dec_dx = p.A * (2.0 / x - (2.0 * x + p.b) / X -
                            p.b * p.x0 / denom * (2.0 / (x - p.x0) - (2.0 * x + p.b) / X));
        };

        vwn_ec(x, vwn_param[0], ec0, dec0_dx);
        vwn_ec(x, vwn_param[1], ec1, dec1_dx);

        double ec = ec0 + (ec1 - ec0) * z2;
        double dec_dx = dec0_dx + (dec1_dx - dec0_dx) * z2;
        double vc = ec - rs / 3.0 * dec_dx / (2.0 * x);

        double rho13 = std::pow(r, 1.0 / 3.0);
        double ex = -Cx * r * rho13;
        double vx = -4.0 / 3.0 * Cx * rho13;

        exc[i] = ex + r * ec;
        vxc[i] = vx + vc;
    }
}

void lda_exc_vxc(int n, const double *rho, double *exc, double *vxc)
{
    lda_exc_vxc_impl(n, rho, exc, vxc, 0.0); // 默认顺磁
}
double compute_exc_energy(int ngrid,
                          const double *weights,
                          const double *rho)
{
    double *exc = (double *)malloc(sizeof(double) * ngrid);
    double *vxc = (double *)malloc(sizeof(double) * ngrid);

    lda_exc_vxc(ngrid, rho, exc, vxc);

    double exc_sum = 0.0;
    for (int g = 0; g < ngrid; ++g)
        exc_sum += weights[g] * exc[g];

    free(exc);
    free(vxc);
    return exc_sum;
}
void build_vxc_matrix(int nao, int ngrid,
                      const double *ao, // shape (ngrid, nao)
                      const double *weights,
                      const double *rho,
                      double *vxc_mat) // shape (nao, nao)
{
    double *exc_buf = (double *)malloc(sizeof(double) * ngrid);
    double *vxc_buf = (double *)malloc(sizeof(double) * ngrid);

    lda_exc_vxc(ngrid, rho, exc_buf, vxc_buf);

    // vxc_mat[i,j] = Σ_g w[g] * vxc[g] * ao[g,i] * ao[g,j]
    std::memset(vxc_mat, 0, sizeof(double) * nao * nao);

    for (int g = 0; g < ngrid; ++g)
    {
        double wv = weights[g] * vxc_buf[g];
        for (int i = 0; i < nao; ++i)
        {
            double aoi = ao[g * nao + i];
            for (int j = 0; j < nao; ++j)
            {
                vxc_mat[i * nao + j] += wv * aoi * ao[g * nao + j];
            }
        }
    }
    free(exc_buf);
    free(vxc_buf);
}

static void eri_entries_to_dense_4d(const std::vector<ERIEntry> &E, int nao, std::vector<double> &eri4)
{
    eri4.assign(size_t(nao) * nao * nao * nao, 0.0);
    for (const auto &e : E)
    {
        size_t idx = (((size_t)e.i * nao + e.j) * nao + e.k) * nao + e.l;
        eri4[idx] = e.v;
    }
}

// 可选：从 entries 直接累加 J（避免构建 4D 张量），四重 for 的展开在 entries 上
static void accumulate_J_from_entries(int nao, const std::vector<ERIEntry> &E,
                                      const double *D, double *J_out)
{
    std::memset(J_out, 0, sizeof(double) * nao * nao);
    for (const auto &e : E)
    {
        // J_mn += D_ls * (mn|ls)；这里 entries 存的是 (i j k l v)
        const int m = e.i, n = e.j, l = e.k, s = e.l;
        J_out[m * nao + n] += D[l * nao + s] * e.v;
    }
}

// 轨道占据：闭壳层 -> 每个空间轨道 2 电子
static int num_occ_from_nelec(int nelec) { return nelec / 2; }

// 对称正交化 X = S^{-1/2}
static Eigen::MatrixXd symmetric_orthogonalizer(const Eigen::MatrixXd &S)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd s = es.eigenvalues();
    Eigen::MatrixXd U = es.eigenvectors();
    Eigen::VectorXd s_inv_sqrt = s.array().inverse().sqrt();
    return U * s_inv_sqrt.asDiagonal() * U.transpose();
}

// KS diagonalize: 在正交基中解本征，再还原 C
static void diagonalize_ks(const Eigen::MatrixXd &F, const Eigen::MatrixXd &X,
                           Eigen::VectorXd &eps, Eigen::MatrixXd &C)
{
    Eigen::MatrixXd Fp = X.transpose() * F * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
    eps = es.eigenvalues();
    Eigen::MatrixXd Cp = es.eigenvectors();
    C = X * Cp;
}

// 构造密度矩阵 D = 2 * C_occ * C_occ^T
static Eigen::MatrixXd build_density(const Eigen::MatrixXd &C, int nocc)
{
    Eigen::MatrixXd Cocc = C.leftCols(nocc);
    return 2.0 * (Cocc * Cocc.transpose());
}

// 迹 Tr[AB]
static double trace_of_product(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
    return (A.cwiseProduct(B.transpose())).sum();
}

// ============== 主流程：RKS-LDA SCF ==============
struct SCFOptions
{
    int max_iter = 100;
    double e_thresh = 1e-9;
    double d_thresh = 1e-6;
    double density_mixing = 0.2; // 线性密度混合
};

struct SCFResult
{
    double Etot;
    double Eelec;
    double Exc;
    double Ehartree;
    Eigen::VectorXd eps; // KS 本征值
    Eigen::MatrixXd C;   // MO 系数
    Eigen::MatrixXd D;   // 收敛密度
    int niter;
    bool converged;
};

SCFResult rks_scf_run(const Eigen::MatrixXd &S,
                      const Eigen::MatrixXd &Hcore,
                      const std::vector<ERIEntry> &eri_entries, // 五列 entries
                      const Eigen::VectorXd &weights,
                      const std::vector<double> &AO_rowmajor, // ngrid*nao
                      int nelec, int ngrid, int nao,
                      double Enuc,
                      const SCFOptions &opts = {})
{
    // 预处理
    const int nocc = num_occ_from_nelec(nelec);
    Eigen::MatrixXd X = symmetric_orthogonalizer(S);

    // 初猜：Core Hamiltonian 密度（或零密度）
    Eigen::VectorXd eps;
    Eigen::MatrixXd C, D = Eigen::MatrixXd::Zero(nao, nao);

    {
        Eigen::MatrixXd F0 = Hcore;
        diagonalize_ks(F0, X, eps, C);
        D = build_density(C, nocc);
    }

    // ERI（可选：若体系很小可转为 4D 密集，以复用你写的 build_coulomb_matrix）
    // 这里我们走“直接从 entries 累加 J”的路径，省内存：
    std::vector<double> Jbuf(nao * nao, 0.0);

    // 用到的裸指针（给你已有的 C 函数）
    std::vector<double> Dbuf(nao * nao, 0.0);
    auto eigen_to_rowmajor = [&](const Eigen::MatrixXd &M, std::vector<double> &out)
    {
        out.resize(size_t(nao) * size_t(nao));
        for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++)
                out[i * nao + j] = M(i, j);
    };

    // 网格相关缓冲
    std::vector<double> rho(ngrid, 0.0);
    std::vector<double> vxc_mat_buf(nao * nao, 0.0);

    double E_last = 0.0;
    bool conv = false;
    int it = 0;
    double Exc_val = 0.0, Ehartree = 0.0, Eelec = 0.0;

    for (it = 1; it <= opts.max_iter; ++it)
    {
        // 1) ρ(g)
        eigen_to_rowmajor(D, Dbuf);
        get_rho(nao, ngrid, Dbuf.data(), AO_rowmajor.data(), rho.data());

        // 2) V_xc 矩阵、E_xc
        build_vxc_matrix(nao, ngrid, AO_rowmajor.data(), weights.data(), rho.data(), vxc_mat_buf.data());
        Exc_val = compute_exc_energy(ngrid, weights.data(), rho.data());
        Eigen::MatrixXd Vxc = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vxc_mat_buf.data(), nao, nao);

        // 3) J（库仑矩阵）: 直接从 entries 累加（内存友好）
        std::fill(Jbuf.begin(), Jbuf.end(), 0.0);
        accumulate_J_from_entries(nao, eri_entries, Dbuf.data(), Jbuf.data());
        Eigen::MatrixXd J = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(Jbuf.data(), nao, nao);

        // 4) Fock(KS) = Hcore + J + Vxc
        Eigen::MatrixXd F = Hcore + J + Vxc;

        // 5) 对角化，构造新密度
        diagonalize_ks(F, X, eps, C);
        Eigen::MatrixXd Dnew = build_density(C, nocc);

        // 6) 线性混合
        Eigen::MatrixXd Dmix = opts.density_mixing * Dnew + (1.0 - opts.density_mixing) * D;

        // 7) 能量：E = Tr[D Hcore] + 0.5 Tr[D J] + E_xc + E_nuc
        Ehartree = 0.5 * trace_of_product(Dmix, J);
        Eelec = trace_of_product(Dmix, Hcore) + Ehartree + Exc_val;
        double Etot = Eelec + Enuc;

        // 8) 收敛判据
        double dE = std::abs(Etot - E_last);
        double dD = (Dmix - D).norm();
        std::cout << "SCF iter " << it
                  << "  Etot = " << std::setprecision(12) << Etot
                  << "  dE = " << dE
                  << "  ||dD|| = " << dD << std::endl;

        if (it > 1 && dE < opts.e_thresh && dD < opts.d_thresh)
        {
            conv = true;
            D = std::move(Dmix);
            E_last = Etot;
            break;
        }

        // 下一轮
        D = std::move(Dmix);
        E_last = Etot;
    }

    SCFResult res;
    res.Etot = E_last;
    res.Eelec = Eelec;
    res.Exc = Exc_val;
    res.Ehartree = Ehartree;
    res.eps = std::move(eps);
    res.C = std::move(C);
    res.D = std::move(D);
    res.niter = it;
    res.converged = conv;
    return res;
}

// ====================== 以下为测试/示例代码 ======================
int main()
{
    // 读入
    Eigen::MatrixXd S = load_matrix_txt("S.txt");
    Eigen::MatrixXd Hcore = load_matrix_txt("Hcore.txt");
    double Enuc = load_scalar_txt("Enuc.txt");
    int nao_chk = 0;
    auto eri_entries = load_eri_entries("ERI.txt", nao_chk);
    if (nao_chk != S.rows())
    {
        std::cerr << "NAO mismatch in ERI vs S/Hcore\n";
        return 1;
    }
    Eigen::VectorXd weights = load_vector_txt("weights.txt");
    int ngrid = 0, nao = 0;
    std::vector<double> AO;
    load_AO_txt("AO.txt", ngrid, nao, AO);
    if (nao != S.rows())
    {
        std::cerr << "NAO mismatch in AO vs S/Hcore\n";
        return 1;
    }
    // 电子数
    int nelec = 8; // H2O

    // 选项
    SCFOptions opts;
    opts.max_iter = 100;
    opts.e_thresh = 1e-9;
    opts.d_thresh = 1e-6;
    opts.density_mixing = 0.3;

    // 运行
    auto res = rks_scf_run(S, Hcore, eri_entries, weights, AO,
                           nelec, ngrid, nao, Enuc, opts);

    std::cout << "\nConverged: " << std::boolalpha << res.converged
              << "  in " << res.niter << " iters\n";
    std::cout << "Etot = " << std::setprecision(12) << res.Etot << "\n";
    std::cout << "Eelec = " << res.Eelec << "  (Exc=" << res.Exc
              << ", Eh=" << res.Ehartree << ")\n";
    return 0;
}