//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_QP_H
#define QCQP_QP_H

#include "utils.h"
#include <queue>
#include <stack>
#include <map>
#include <ctime>

class QP {
public:
    // (n,d): size of x
    // m: # of constrs,
    // N: size of homogenized matrices, N:=(n+1)^2
    const int n, m, d;
    const int N;
    // naming convention for matrices
    // ()h -> homogenized
    // ()c -> convexified
    // objective Q∙Y + q∙x
    //  and homogenized Qh
    eigen_matrix Q;
    eigen_array q;
    eigen_matrix Qh;
    // diagonal constraint
    //  diag(Y) <= x
    // homogenized by Qd
    // note this is only needed by SDP
    std::vector<eigen_matrix> Qd;
    // constraints A_i∙Y + a_i∙x <= b_i
    //  and homogenized Ah_i
    std::vector<eigen_matrix> Ah;
    std::vector<eigen_matrix> Ar;
    std::vector<eigen_array> ar;
    eigen_array br;
    bool verbose = false; // debug only

    // spectral decompositions
    // stored in eigen-solver
    // we create A by A[0] = Q, A[1:] = Ar
    //  so that the matrices are unified
    //  A in R(m + 1, n, n)
    std::vector<Eigen::SelfAdjointEigenSolver<eigen_matrix>> vec_es;
    std::vector<eigen_matrix> Ac;
    std::vector<std::vector<int>> Ac_rows;
    std::vector<std::vector<int>> Ac_cols;
    std::vector<std::vector<double>> Ac_vals;

    std::vector<eigen_array> Dc;
    std::vector<eigen_matrix> A;
    std::vector<eigen_array> a;
    std::vector<double> b;
    eigen_matrix V;

    // functions
    QP(int n, int m, int d, double *Q_data, double *q_data);

    QP(int n, int m, int d, double *Q_data, double *q_data,
       double *A_data, double *a_data, double *b_data);

    void create_diag_Q(int n, int m);

    void show();

    double inhomogeneous_obj_val(double *x) const;

    void setup();
    void convexify(int method = 0);
};

class Backend {

public:
    explicit Backend(QP &qp);
};

// QP UTIL FUNCS.
eigen_matrix homogenize_quadratic_form(eigen_matmap A, eigen_arraymap a);
eigen_matrix homogenize_quadratic_form(eigen_matmap A, eigen_arraymap a, double b);
#endif //QCQP_QP_H
