//
// Created by C. Zhang on 2021/9/5.
//

#ifndef QCQP_RESULT_H
#define QCQP_RESULT_H

#include "utils.h"
#include "qp.h"

class Result {
public:
    const int n, m, d;
    double *x{};
    double *z{};
    double *X{};
    double *y{};
    double *Y{};
    double *D{};
    double *S{};
    double *Dd{};
    double *Sd{};
    int ydim{};
    // eigen representations
    //  for some backend, may not be available
    eigen_const_matmap Xm;
    eigen_const_matmap Ym;
    // residual: X - xx^T
    eigen_matrix Res;
    // info
    // objectives
    // this primal value is simply extract the vector from homonegenized problem
    //  I do not implement any primal heuristic here.
    double primal = 1e6;
    double relax = -1e6;
    double primal_feas, dual_feas = 0;
    int iterations = 0;


    Result(int n, int m, int d);

    ~Result();

    void save_to_X(double *X_);

    void save_to_Y(double *Y_);

    int show(bool x_only = false);
};


#endif //QCQP_RESULT_H
