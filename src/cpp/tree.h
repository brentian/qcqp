//
// Created by chuwen on 2021/9/9.
//

#ifndef QCQP_TREE_H
#define QCQP_TREE_H

#include <queue>
#include <stack>
#include "cut.h"
#include "branch.h"
#include "result.h"
#include "qp.h"
#include "utils.h"


//template<typename T> class Node {
//public:
//
//    Node(long id, QP &qp, // no dfts.
//         long parent_id = -1, long depth = 0,
//         double bound = 0.0, double primal_val = 0.0,
//         double abs_time = 0.0, double create_time = 0.0, double solve_time = 0.0
//    );
//
//    T p;
//    long id;
//    long parent_id;
//    long depth;
//    double obj_bound;
//    double primal_val;
//    double abs_time;
//    double create_time;
//    double solve_time;
//};


class Node {
public:
    Node() = default;
//    Node(long id, QP &qp, // no dfts.
//         long parent_id = -1, long depth = 0,
//         double bound = 0.0, double primal_val = 0.0,
//         double abs_time = 0.0, double create_time = 0.0, double solve_time = 0.0
//    );

//    Backend p;
    long id;
    long parent_id;
    long depth;
    double obj_bound;
    double primal_val;
    double abs_time;
    double create_time;
    double solve_time;
    CutPool ct;

    void attach_cuts(CutPool another_cp);
};


class Tree {
public:
//    std::stack<Node> queue;

    virtual int run(QP &qp);
};


#endif //QCQP_TREE_H

