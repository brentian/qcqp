////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2022.                                                                             /
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Created by C. Zhang on 2022/3/10.
//

#ifndef QCQP_BB_COPT_H
#define QCQP_BB_COPT_H

#include "qp.h"
#include "utils.h"
#include "copt.h"
#include "cut.h"
#include "tree.h"
#include "branch.h"
#include "bg_copt.h"

class Node_COPT : public Node {
public:
    Bg_COPT p;
    Pr_COPT pr;
    bool bool_solved = false;
    bool bool_setup = false;

    Node_COPT(long id, QP &qp,// no dfts.
              long parent_id = -1, long depth = 0,
              double parent_bound = 1e6,
              double bound = 0.0, double primal_val = 0.0
    );

    ~Node_COPT() = default;

    void create_problem(copt_env *env, CutPool &cp, Bound &bound) {
      time_opt_start=std::chrono::steady_clock::now();
      // push cuts
#if QCQP_BRANCH_DBG
      p.create_problem(env, bound, false, true);
#else
      p.create_problem(env, bound);
#endif
      bool_setup = true;
    }

    void optimize() {
      p.optimize();
      bool_solved = true;
      time_opt_end = std::chrono::steady_clock::now();
      time_solve = time_opt_end - time_opt_start;
    }

    void extract_solution();

    Result_COPT get_solution() const {
      return p.get_solution();
    }

    // primal method
    void create_primal(copt_env *env, Bound &bound) {
      pr.create_trs_copt(env, p, bound, p.r);
    }

    void optimize_primal(copt_env *env, QP &qp, Bound &bound) {
      pr.optimize(env, qp, p, bound);
    }

    Result_COPT get_primal_solution() const {
      return pr.r;
    }

};


class Tree_COPT : public Tree<Node_COPT, Result_COPT> {
public:

    Tree_COPT() = default;

    int run(QP &qp, Bound &root_b, copt_env *env, Params &param);

    int iter(Node_COPT &node, Params &param, QP &qp, copt_env *env, long itern);

    int iter_primal(Node_COPT &node, Bound &bound, Params &param, QP &qp, copt_env *env, long itern);
};

class Branch_COPT : public Branch<Result_COPT> {
public:
};

#endif //QCQP_BB_COPT_H
