//
// Created by C. Zhang on 2021/9/16.
//

#include "bb_dsdp.h"

Node_DSDP::Node_DSDP(long id, QP &qp, long parent_id, long depth, double bound, double primal_val, double abs_time,
                     double create_time, double solve_time) : p(qp) {
    this->id = id;
    this->parent_id = parent_id;
    this->depth = depth;
    this->obj_bound = bound;
    this->primal_val = primal_val;
    // timers
    this->abs_time = abs_time;
    this->create_time = create_time;
    this->solve_time = solve_time;
}

void Node_DSDP::extract_solution() {
    p.extract_solution();
}

int Tree_DSDP::run(QP &qp) {
    Bound root_b(qp.n);
    Node_DSDP root(0, qp);
    root.create_problem();
    root.optimize();
    root.extract_solution();
    auto root_r = root.get_solution();

    queue.push(root);
    while (!queue.empty()) {
        auto node = queue.top();
        if (!node.p.bool_solved) {
            node.create_problem();
            node.optimize();
            node.extract_solution();
        }
        auto node_r = node.get_solution();
        // evaluate
        auto Res = node_r.Res;

        fprintf(stdout,
                "time: %.2f # %ld %ld "
                "infeas: %.3e primal %.4f bound %.4f "
                "gap %.4f [lb %.3f, ub %.3f]",
                node.abs_time, node.id, node.depth,
                0.1, 0.1, 0.1,
                0.1, 0.1, 0.1
        );

    }
}
