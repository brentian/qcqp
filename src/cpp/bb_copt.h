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
    QP_COPT p;
    bool bool_solved = false;
    bool bool_setup = false;

    Node_COPT(long id, QP &qp,// no dfts.
              long parent_id = -1, long depth = 0,
              double parent_bound = 1e6,
              double bound = 0.0, double primal_val = 0.0
    );

    ~Node_COPT() {
      COPT_DeleteProb(&p.prob);
    }

    void create_problem(copt_env *env, CutPool &cp, Bound &bound) {
      time(&time_opt_start);
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
      time(&time_opt_end);
      time_solve = difftime(time_opt_end, time_opt_start);
    }

    void extract_solution();

    Result_COPT get_solution() const {
      return p.get_solution();
    }

};


class Tree_COPT : public Tree<Node_COPT, Result_COPT> {
public:
    time_t timer{};

    Tree_COPT() = default;

    int run(QP &qp, Bound &root_b, copt_env *env, Params &param);

    int iter(Node_COPT &node, Params &param, QP &qp, copt_env *env, long itern);

    std::pair<long, double> fetch_next() {
      auto kv = get_max(map_ub);
      return kv;
    }
};

class Branch_COPT : public Branch<Result_COPT> {
public:
};

#endif //QCQP_BB_COPT_H
