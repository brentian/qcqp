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

    Result_DSDP(int n, int m, int d) :
            Result(n, m, d) {
    }

    void construct_init_point(Result_DSDP &r, double lambda = 0.99, int pool_size = 0);

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
    // backend
    DSDP p;
    SDPCone sdpcone;
    BCone bcone;

    //
    bool solved = false;
    const int m_with_cuts{};
    const int m;

    explicit QP_DSDP(QP &qp) : qp(qp), r(qp.n, qp.m, qp.d), m(qp.m), m_with_cuts(cp.size() + qp.m) {
    }

    ~QP_DSDP() { DSDPDestroy(p); }

    void create_problem(bool solve = false, bool verbose = true, bool use_lp_cone = false);

    void assign_initial_point(Result_DSDP &r, bool dual_only);

    void solve_sdpa_p(bool verbose = false);

    void extract_solution();

    void print_sdpa_formatted_solution();

    Result_DSDP get_solution();
};

#endif //QCQP_BG_DSDP_H
