//
// Created by C. Zhang on 2021/7/25.
//

#ifndef QCQP_BG_SDPA_H
#define QCQP_BG_SDPA_H

#include "qp.h"
#include "utils.h"
#include "sdpa_call.h"
#include "cut.h"


class Result_SDPA : public Result {
public:

    Result_SDPA(int n, int m, int d) :
            Result(n, m, d) {
    }

    void construct_init_point(Result_SDPA &r, double lambda = 0.99, int pool_size = 0);

    void check_solution(QP &qp);

    void show();
};

struct Bound_SDPA : Bound {

    double *xlb{};
    double *xub{};

    Bound_SDPA(int n) {
        xlb = new double[n]{0.0};
        xub = new double[n]{0.0};
    }
};

struct RLT_SDPA : RLT {

    void create_from_bound(int n, int i, int j, Bound_SDPA &bd);

    void create_from_bound(int n, int i, int j, double li, double ui, double lj, double uj);
};

typedef std::vector<Cut> CutPool;

class QP_SDPA {
private:
    QP &qp;
public:
    Result_SDPA r;
    CutPool cp;
    SDPA p = SDPA();
    bool solved = false;
    int ydim;
    int m;

    explicit QP_SDPA(QP &qp) : qp(qp), r(qp.n, qp.m, qp.d) {
    }

    ~QP_SDPA() { p.terminate(); }

    void create_sdpa_p(bool solve = false, bool verbose = true);

    void assign_initial_point(Result_SDPA &r, bool dual_only);

    void solve_sdpa_p(bool verbose = false);

    void extract_solution();

    void print_sdpa_formatted_solution();

    Result_SDPA get_solution();
};


#endif //QCQP_BG_SDPA_H
