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

    Result_COPT(int n, int m, int d);

    void construct_init_point(Result_COPT &r, double lambda = 0.99, int pool_size = 0);
};

class QP_COPT : public Backend {
    explicit QP_COPT(QP &qp);

    ~QP_COPT();

    void setup();

    void create_problem(bool solve = false, bool verbose = false, bool use_lp_cone = false);

    void assign_initial_point(Result_COPT &r_another, bool dual_only) const;

    void extract_solution();

    Result_COPT get_solution() const;

    void optimize();
};


#endif //QCQP_BG_COPT_H
