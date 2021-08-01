//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_BG_SDPA_H
#define QCQP_BG_SDPA_H

#include "qp.h"
#include "utils.h"
#include "sdpa_call.h"


struct Result_SDPA : public Result {

    Result_SDPA(int i, int i1, int i2);

    const double *y{};
    const double *Y{};
    const double *D{};
    const double *S{};


    Result_SDPA construct_init_point(double lambda = 0.99);
    void show();
};

class QP_SDPA {

public:
    QP qp;
    SDPA p;
    Result_SDPA r;

    bool solved = false;


    explicit QP_SDPA(QP &qp);

    ~QP_SDPA() { p.terminate(); }

    void create_sdpa_p(bool solve = false, bool verbose = true);

    void assign_initial_point(double *X_init, const double *y_init, const double *Y_init, bool dual_only = true);
    void assign_initial_point(eigen_const_matmap X, eigen_const_arraymap y, eigen_const_matmap Y, bool dual_only);

    void solve_sdpa_p(bool verbose = false);

    void extract_solution();

    };


#endif //QCQP_BG_SDPA_H
