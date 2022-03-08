// branch and bound for QCQP
//  using SDP (DSDP) as backend.

#ifndef QCQP_BB_SDPA_H
#define QCQP_BB_SDPA_H

#include "qp.h"
#include "utils.h"
#include "dsdp5.h"
#include "cut.h"
#include "tree.h"
#include "bg_dsdp.h"
#include "bg_dsdp_cut.h"

class Node_DSDP : public Node {
public:
    QP_DSDP p;
    bool bool_solved = false;
    bool bool_setup = false;

    Node_DSDP(long id, QP &qp, // no dfts.
              long parent_id = -1, long depth = 0,
              double parent_bound = 1e6,
              double bound = 0.0, double primal_val = 0.0
    );


    void create_problem(CutPool &cp) {
        time(&time_opt_start);
        // push cuts
        p.cp = cp;
#if QCQP_BRANCH_DBG
        p.create_problem(false, true);
#else
        p.create_problem();
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

    Result_DSDP get_solution() const {
        return p.get_solution();
    }

};

class Params {
public:
    //        bool verbose=false;
    bool warmstart = true;
    double tolgap = 1e-4;
    double tolfeas = 1e-4;
    double lb = -1e6;
    double ub = -1e6;
    double gap = 1e6;
};

class Tree_DSDP : public Tree<Node_DSDP, Result_DSDP> {
public:
    time_t timer;

    int run(QP &qp, Params &param);

    int iter(Node_DSDP &node, Params &param, QP &qp);

    template<typename KeyType, typename ValueType>
    std::pair<KeyType, ValueType> fetch_next();
};

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType> &x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [](const pairtype &p1, const pairtype &p2) {
        return p1.second < p2.second;
    });
}


#endif //QCQP_BB_SDPA_H
