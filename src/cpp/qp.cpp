//
// Created by C. Zhang on 2021/7/25.
//

#include "qp.h"


QP::QP(int n, int m, int d, double *Q_data, double *q_data)
        : n(n), m(m), d(d) {
    Q = eigen_matrix(n, n);
    q = eigen_array(n);
    for (int i = 0; i <= n - 1; i++) {
        q(i) = q_data[i];
        for (int j = 0; j <= n - 1; j++)
            Q(i, j) = Q_data[i * n + j];
    }
    Eigen::Map<Eigen::RowVectorXd> vtmp(q.data(), n);
    Qhfull = eigen_matrix::Zero(n + 1 + n + m, n + 1 + n + m);
    Qh = eigen_matrix::Zero(n + 1, n + 1);
    Qh << Q, q / 2, vtmp / 2, eigen_matrix::Zero(1, 1);
    Qhfull.block(0, 0, n + 1, n + 1) = Qh;
    create_diag_Q(n, m);
}

void QP::create_diag_Q(int n, int m) {
    Qdiag = std::vector<eigen_matrix>(n, eigen_matrix::Zero(n + 1 + n + m, n + 1 + n + m));
    bdiag = std::vector<double>(n, 0);
    int i = 0;

    for (auto &Qi:Qdiag) {
        auto *e = new double[n]{0};
        auto *ee = new double[n * n]{0};
        e[i] = -0.5;
        ee[i * n + i] = 1;
        Eigen::Map<Eigen::RowVectorXd> rtmp(e, n);
        Eigen::Map<eigen_matrix> vtmp(e, n, 1);
        Eigen::Map<eigen_matrix> btmp(ee, n, n);
        eigen_matrix Qtmp(n + 1, n + 1);
        Qtmp << btmp, vtmp, rtmp, eigen_matrix::Zero(1, 1);
        Qi.block(0, 0, n + 1, n + 1) = Qtmp;
        Qi.block(n + 1, n + 1, n, n) = btmp;
        i++;
    }
    if (verbose) {
        for (auto &Qi:Qdiag) {
            std::cout << "Q\n";
            std::cout << Qi << std::endl;
        }
    }
}

