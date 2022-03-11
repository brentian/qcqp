////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022.                                                                             /
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Created by C. Zhang on 2022/3/8.
//

#ifndef QCQP_BG_COPT_H
#define QCQP_BG_COPT_H

#include "qp.h"
#include "utils.h"
#include "cut.h"
#include "tree.h"
#include "copt.h"

class Result_COPT : public Result {
public:
    double r0;
    eigen_array x;
    eigen_array z;
    eigen_array xall;
    eigen_array y;
    Result_COPT(int n, int m, int d);

    void construct_init_point(Result_COPT &r, double lambda = 0.99, int pool_size = 0);
};

class QP_COPT : public Backend {
public:

    QP &qp;
    copt_prob *prob = nullptr;
    Result_COPT r;
    CutPool cp;

    int ncol;
    int ydim;
    explicit QP_COPT(QP &qp);

    ~QP_COPT() {
      COPT_DeleteProb(&prob);
    }

    void setup();

    void create_problem(copt_env *env, Bound & bound, bool solve = false, bool verbose = false);

    void assign_initial_point(Result_COPT &r_another, bool dual_only) const;

    void extract_solution();

    Result_COPT get_solution() const;

    void optimize();
};


#endif //QCQP_BG_COPT_H
