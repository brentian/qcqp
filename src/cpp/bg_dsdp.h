//
// Created by chuwen on 2021/8/30.
//

#ifndef QCQP_BG_DSDP_H
#define QCQP_BG_DSDP_H
#include "qp.h"
#include "utils.h"
#include "dsdp5.h"
#include "cut.h"
class Result_DSDP : public Result {
public:
    eigen_const_matmap Ym;

    Result_DSDP(int n, int m, int d) :
    Result(n, m, d),
    Ym(nullptr, n + 1, n + 1) {
    }

    void save_to_Y(double *Y_) {
        Y = Y_;
        new(&Ym) eigen_const_matmap(Y_, n + 1, n + 1);
    };

    void construct_init_point(Result_DSDP &r, double lambda = 0.99, int pool_size=0);

    void check_solution(QP &qp);

    void show();
};

struct Bound_DSDP : Bound {

    double *xlb{};
    double *xub{};

    Bound_DSDP(int n) {
        xlb = new double[n]{0.0};
        xub = new double[n]{0.0};
    }
};

struct RLT_DSDP : RLT {

    void create_from_bound(int n, int i, int j, Bound_DSDP &bd);

    void create_from_bound(int n, int i, int j, double li, double ui, double lj, double uj);
};

typedef std::vector<Cut> CutPool;

class QP_DSDP {
private:
    QP &qp;
public:
    Result_DSDP r;
    CutPool cp;
    DSDP p = DSDP();
    bool solved = false;
    int m_with_cuts;
    int m;
    explicit QP_DSDP(QP &qp) : qp(qp), r(qp.n, qp.m, qp.d) {
    }

//    ~QP_DSDP() { p.terminate(); }

    void create_problem(bool solve = false, bool verbose = true);

    void assign_initial_point(Result_DSDP &r, bool dual_only);

    void solve_sdpa_p(bool verbose = false);

    void extract_solution();
    void print_sdpa_formatted_solution();

    Result_DSDP get_solution();
};

#endif //QCQP_BG_DSDP_H
