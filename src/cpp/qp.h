//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_QP_H
#define QCQP_QP_H

#include "utils.h"

class QP {
public:
    const int n, m, d;
    eigen_array q;
    eigen_matrix Q;
    eigen_matrix Qh;
    eigen_matrix Qhfull;
    std::vector<eigen_matrix> Qdiag;
    std::vector<double> bdiag;
    bool verbose = false;

    QP(int n, int m, int d, double *Q_data, double *q_data);

    void create_diag_Q(int n, int m);


};

struct Result {
    const int n, m, d;
    double *x{};
    double *X{};
    Result(int n, int m, int d): n(n), m(m), d(d){};
};
#endif //QCQP_QP_H
