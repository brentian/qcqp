//
// Created by C. Zhang on 2021/9/16.
//


#include "bb_dsdp.h"


Node_DSDP::Node_DSDP(
        long id, QP &qp, long parent_id, long depth,
        double parent_bound,
        double bound, double primal_val
) : p(qp) {
    this->id = id;
    this->parent_id = parent_id;
    this->depth = depth;
    this->obj_bound = bound;
    this->primal_val = primal_val;
    this->parent_bound = parent_bound;
    // timers
    this->create_time = time(&create_time);
}

void Node_DSDP::extract_solution() {
    p.extract_solution();
    obj_bound = p.bound;
    primal_val = p.primal;
}

int Tree_DSDP::run(QP &qp) {
    Bound root_b(qp.n);
    Node_DSDP root(0, qp);

    cut_queue[0] = root.p.cp;
    bound_queue[0] = Bound(root_b);
    ub_queue[0] = 1e6;
    double ub = -1e6;
    double lb = -1e6;
    double gap;

    time(&timer);
    queue.insert(std::pair<long, Node_DSDP>(0, root));
    auto bn = &root;
    while (!queue.empty()) {
        std::pair<long, double> next_kv = fetch_next<long, double>();
        long next_id = next_kv.first;
        auto node = queue.at(next_id);
        auto current_cuts = cut_queue[node.id];
        auto current_bound = bound_queue[node.id];
        // clear whatever will not be used.
        queue.erase(next_id);
        bound_queue.erase(node.id);
        cut_queue.erase(node.id);
        ub_queue.erase(node.id);

        if (!node.bool_solved) {
            node.create_problem(current_cuts);
            node.optimize();
            node.extract_solution();
        }
        auto node_r = node.get_solution();
        // evaluate
        auto Res = node_r.Res;
        double infeas = Res.maxCoeff();
        double bound = node.obj_bound;
        double primal = node.primal_val;

        // timers
        auto solve_time = difftime(node.opt_end_time, timer);
        // update lower bound by primal solution
        if (primal > lb) {

            lb = primal;
            bn = &node;
        }
        // update upper bound by sdp relaxation
        ub = std::max(bound, next_kv.second);
        gap = (ub - lb) / lb;
        // status report
        fprintf(stdout,
                "time: %.2f #%ld@%ld "
                "infeas: %.3e primal %.4f bound %.4f "
                "gap %.4f [%.3f, %.3f] \n",
                solve_time, node.id, node.depth,
                infeas, primal, bound,
                gap, lb, ub
        );
        // prunes
        if (bound < lb) {
            // bound prune.
            fprintf(stdout, "prune #%ld \n", node.id);
            continue;
        }
        if (infeas < 1e-4) {
            // primal solution is feasible rank-1 solution
            fprintf(stdout, "prune #%ld by feasibility\n", node.id);
            continue;
        }
        if (gap <= 1e-4) {
            // prune by gap
            fprintf(stdout, "prune #%ld by gap\n", node.id);
            break;
        }
        // branch
        Branch br(node_r);
        br.imply_bounds(current_bound);
        // create child
        long child_depth = node.depth + 1;
        long left_id = node.id + 1;
        long right_id = node.id + 2;

        // create cuts
        auto lc = CutPool(current_cuts);
        auto rc = CutPool(current_cuts);
        // todo, consider add more
        lc.push_back(RLT_DSDP::create_from_branch(br, 0));
        rc.push_back(RLT_DSDP::create_from_branch(br, 1));

        // nodes
        auto left_node = Node_DSDP(
                left_id, qp, node.id, child_depth,
                bound, 0.0, 0.0
        );
        auto right_node = Node_DSDP(
                right_id, qp, node.id, child_depth,
                bound, 0.0, 0.0
        );
        // update containers
        bound_queue[left_id] = br.left_b;
        bound_queue[right_id] = br.right_b;
        cut_queue[left_id] = lc;
        cut_queue[right_id] = rc;
        ub_queue[left_id] = bound;
        ub_queue[right_id] = bound;

        queue.insert(std::pair<long, Node_DSDP>(left_id, left_node));
        queue.insert(std::pair<long, Node_DSDP>(right_id, right_node));
    }

    return 1;
}

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> Tree_DSDP::fetch_next() {
    auto kv = get_max(ub_queue);
    return kv;
}

