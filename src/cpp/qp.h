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
public:
    const int n, m, d;
    const double *x{};
    const double *X{};
    const double *y{};
    const double *Y{};
    const double *D{};
    const double *S{};
// eigen representations
//  for some backend, may not be available
//    eigen_const_arraymap xm;
    eigen_const_matmap Xm;
//    eigen_const_matmap Ym;
//    eigen_const_arraymap Dm;
//    eigen_const_arraymap Sm;

    Result(int n, int m, int d) :
            n(n), m(m), d(d), Xm(X, n+1, n+1){
    };
    void save_to_X(const double *X_){
        X = X_;
        new (&Xm) eigen_const_matmap(X_, n+1, n+1);
    };
};

#endif //QCQP_QP_H
