//
// Created by chuwen on 2021/8/30.
//

#ifndef QCQP_BG_DSDP_H
#define QCQP_BG_DSDP_H

#include "qp.h"
#include "utils.h"
#include "dsdp5.h"
#include "cut.h"


#define dbg 1

class Result_DSDP : public Result {
public:
    double r0;

    Result_DSDP(int n, int m, int d);

    void construct_init_point(Result_DSDP &r, double lambda = 0.99, int pool_size = 0);
};

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
    const int n;
    const int m;
    // number of variables (for original)
    int m_only_cuts;
    int m_with_cuts;
    // problem size
    int ndim; // homogeneous
    int n_p_square; // # nnz in the full matrix
    int n_lower_tr; // # nnz in the lower triangular block
    // total number of constrs;
    // - 1 for ynn = 1
    // - n diagonal constr: y[i, i] <= x[i]
    // - m_with_cuts
    //      - m quadratic constr
    //      - cp.size() cuts
    int nvar;

    // dynamic arrays;
    double *_tilde_q_data;
    // value and index for 1-constraint
    double _one_val[1]{1.0};
    int _one_idx[1]{0};
    // value and index for diagonal-constraint
    double _ei_val[2]{0.0};
    int *_ei_idx;
    // value for quadratic constraints
    double *_ah_data;

    explicit QP_DSDP(QP &qp);

    ~QP_DSDP();

    void setup();

    void create_problem(bool solve = false, bool verbose = true, bool use_lp_cone = false);

    void assign_initial_point(Result_DSDP &r_another, bool dual_only);

    void extract_solution();

    Result_DSDP get_solution() const;

    void optimize();
};

#endif //QCQP_BG_DSDP_H
