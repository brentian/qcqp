//
// Created by chuwen on 2021/9/9.
//

#ifndef QCQP_TREE_H
#define QCQP_TREE_H


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
    double parent_bound;
    double obj_bound;
    double primal_val;
    time_t opt_start_time;
    time_t create_time;
    time_t opt_end_time;
    double solve_time;
    // Cut and variable bound
    //    CutPool ct;
    //    Bound bd;
    //    void attach_cuts(CutPool another_cp);
    //    void attach_bound(Bound &another);
    //    CutPool get_cuts();
};


class Tree {
public:
//    std::stack<Node> queue;
    std::map<long, Bound> bound_queue;
    std::map<long, CutPool> cut_queue;
    std::map<long, double> ub_queue;
    long best_node_id = 0;
    long total_nodes = 0;
//    virtual int run(QP &qp) = 0;
};


#endif //QCQP_TREE_H

