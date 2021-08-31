//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_QP_H
#define QCQP_QP_H

#include "utils.h"

class QP {
public:
    // (n,d): size of x
    // m: # of constrs,
    // N: size of homogenized matrices, N:=(n+1)^2
    const int n, m, d;
    const int N;
    // objective Q∙Y + q∙x
    //  and homogenized Qh
    eigen_matrix Q;
    eigen_array q;
    eigen_matrix Qh;
    __unused eigen_matrix Qhfull; // not used
    // diagonal constraint
    //  diag(Y) <= x
    // homogenized by Qd
    std::vector<eigen_matrix> Qd;
    // constraints A_i∙Y + a_i∙x <= b_i
    //  and homogenized Ah_i
    std::vector<eigen_matrix> Ah;
    std::vector<eigen_matrix> A;
    std::vector<eigen_array> a;
    eigen_array b;
    bool verbose = false; // debug only

    QP(int n, int m, int d, double *Q_data, double *q_data);

    QP(int n, int m, int d, double *Q_data, double *q_data,
       double *A_data, double *a_data, double *b_data);

    void create_diag_Q(int n, int m);

    void show();
};

// QP UTIL FUNCS.
eigen_matrix homogenize_quadratic_form(eigen_matmap A, eigen_arraymap a);

class Result {
public:
    const int n, m, d;
    double *x{};
    double *X{};
    double *y{};
    double *Y{};
    double *S{};
    double *Dd{};
    double *Sd{};
    int m_with_cuts;
// eigen representations
//  for some backend, may not be available
//    eigen_const_arraymap xm;
    eigen_const_matmap Xm;
//    eigen_const_matmap Ym;
//    eigen_const_arraymap Dm;
//    eigen_const_arraymap Sm;

    Result(int n, int m, int d) :
            n(n), m(m), d(d), Xm(nullptr, n + 1, n + 1) {
    };

    void save_to_X(double *X_) {
        X = X_;
        new(&Xm) eigen_const_matmap(X_, n + 1, n + 1);
    };
    double *D{};
};

#endif //QCQP_QP_H
