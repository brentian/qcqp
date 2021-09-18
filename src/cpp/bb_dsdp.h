//
// Created by C. Zhang on 2021/9/16.
//

#ifndef QCQP_BB_DSDP_H
#define QCQP_BB_DSDP_H

#include "qp.h"
#include "utils.h"
#include "dsdp5.h"
#include "cut.h"
#include "tree.h"
#include "bg_dsdp.h"
#include "bg_dsdp_cut.h"


//template Node<QP_DSDP> Node_DSDP;
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
        time(&opt_start_time);
        // push cuts
        p.cp = cp;
        p.create_problem();
        bool_setup = true;
    }

    void optimize() {
        p.optimize();
        bool_solved = true;
        time(&opt_end_time);
        solve_time = difftime(opt_end_time, opt_start_time);
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
};

class Tree_DSDP : public Tree<Node_DSDP, Result_DSDP> {
public:
    time_t timer;

    int run(QP &qp, Params &param);

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


#endif //QCQP_BB_DSDP_H
