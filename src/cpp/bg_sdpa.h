//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_BG_SDPA_H
#define QCQP_BG_SDPA_H

#include "qp.h"
#include "utils.h"
#include "sdpa_call.h"


struct Result_SDPA : Result {
    eigen_const_matmap Ym;

    Result_SDPA(int n, int m, int d) :
            Result(n, m, d),
            Ym(Y, n + 1, n + 1) {
    }

    void save_to_Y(const double *Y_) {
        Y = Y_;
        new(&Ym) eigen_const_matmap(Y_, n + 1, n + 1);
    };

    Result_SDPA construct_init_point(double lambda = 0.99);

    void check_solution(QP &qp);
    void show();
};

class QP_SDPA {
private:
    QP qp;
    SDPA p;
    Result_SDPA r;
public:
    bool solved = false;

    explicit QP_SDPA(QP &qp) : qp(qp), r(qp.n, qp.m, qp.d) {

    }

    ~QP_SDPA() { p.terminate(); }

    void create_sdpa_p(bool solve = false, bool verbose = true);

    void assign_initial_point(const double *X_init, const double *y_init, const double *Y_init, bool dual_only = true);

    void assign_initial_point(eigen_const_matmap X, eigen_const_arraymap y, eigen_const_matmap Y, bool dual_only);

    void solve_sdpa_p(bool verbose = false);

    void extract_solution();

    Result_SDPA get_solution();
};


#endif //QCQP_BG_SDPA_H
