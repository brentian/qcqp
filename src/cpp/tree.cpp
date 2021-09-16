//
// Created by chuwen on 2021/9/9.
//

#include "tree.h"

#include <utility>

//template<typename T>
//Node<T>::Node(long id, QP &qp, long parent_id, long depth, double bound, double primal_val, double abs_time,
//              double create_time, double solve_time) : p(qp) {
//    this->id = id;
//    this->parent_id = parent_id;
//    this->depth = depth;
//    this->depth = depth;
//    this->obj_bound = bound;
//    this->primal_val = primal_val;
//    // timers
//    this->abs_time = abs_time;
//    this->create_time = create_time;
//    this->solve_time = solve_time;
//};
void Node::attach_cuts(CutPool another_cp) {
    ct = CutPool(another_cp);
}
