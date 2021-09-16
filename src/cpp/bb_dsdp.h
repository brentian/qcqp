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


//template Node<QP_DSDP> Node_DSDP;
class Node_DSDP : public Node {
public:

    QP_DSDP p;

    Node_DSDP(long id, QP &qp, // no dfts.
              long parent_id = -1, long depth = 0,
              double bound = 0.0, double primal_val = 0.0,
              double abs_time = 0.0, double create_time = 0.0, double solve_time = 0.0
    );

    void create_problem() {
        p.create_problem();
    }

    void optimize() {
        p.optimize();
    }

    void extract_solution();

    Result_DSDP get_solution() const {
        return p.get_solution();
    }
};

class Tree_DSDP : public Tree {
public:
    std::stack<Node_DSDP> queue;

    int run(QP &qp);
};

#endif //QCQP_BB_DSDP_H
