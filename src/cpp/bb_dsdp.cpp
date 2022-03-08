//
// Created by C. Zhang on 2021/9/16.
//


#include "bb_dsdp.h"


Node_DSDP::Node_DSDP(
        long id, QP &qp, long parent_id, long depth,
        double parent_bound,
        double bound, double primal_val
) : Node(), p(qp) {
    this->id = id;
    this->id_parent = parent_id;
    this->depth = depth;
    this->val_rel = bound;
    this->val_prm = primal_val;
    this->val_rel_pa = parent_bound;
    // timers
    this->time_created = time(&time_created);
}

void Node_DSDP::extract_solution() {
    p.extract_solution();
    val_rel = p.r.bound;
    val_prm = p.r.primal;
}

int Tree_DSDP::iter(Node_DSDP &node, Params &param, QP &qp) {
    // prunes
    auto node_id = node.id;
    auto current_cuts = map_cuts[node_id];
    auto current_bound = map_bound[node_id];
    if (node.val_rel_pa < param.lb) {
        // prune by parent
        // fprintf(stdout, "prune #%ld by parent\n", node_id);
        return 0;
    }

    if (!node.bool_solved) {
        node.create_problem(current_cuts);
        // warm-start is only needed if it has to be solved;
        if (param.warmstart and node_id) {
            auto parent_r = map_result.at(node.id_parent);
            auto rws = Result_DSDP(qp.n, qp.m, qp.d);
            rws.construct_init_point(parent_r, 0.99, current_cuts.size());
            node.p.assign_initial_point(rws, true);
        }
        node.optimize();
        node.extract_solution();
    }
    if (node_id and map_num_unsolved_child[node.id_parent] == 0) {
        map_result.erase(node.id_parent);
    }
    auto node_r = node.get_solution();

    // evaluate
    auto Res = node_r.Res;
    double infeas = Res.maxCoeff();
    double relax = node.val_rel;
    double primal = node.val_prm;

    // timers
    auto solve_time = difftime(node.time_opt_end, timer);
    // update lower bound by primal solution
    if (primal > param.lb) {
        param.lb = primal;
        best_node_id = node_id;
        if (!best_result.empty()) {
            best_result.pop();
        }
        best_result.push(node_r);
    }
    // update upper bound by sdp relaxation
    param.ub = node.val_rel_pa;
    param.gap = (param.ub - param.lb) / abs((param.lb) + param.tolfeas);
    int info;
    std::string status;
    if (param.gap <= param.tolgap) {
        info = 1;
        status = "finished";
    } else if (relax < param.lb) {
        // bound prune.
        info = 0;
        status = "D";
    } else if (infeas < param.tolfeas) {
        // primal solution is feasible rank-1 solution
        info = 0;
        status = "P";
    } else {
        status = "";
        info = -1;
    }
    // status report
    gen_status_report(
            solve_time, node_id, node.depth,
            current_cuts.size(), node.p.r.iterations,
            primal, relax, infeas, param.gap,
            param.lb, param.ub, status
    );
    return info;
}


int Tree_DSDP::run(QP &qp, Params &param) {
    Bound root_b(qp.n);
    Node_DSDP root(0, qp);

    print_header();

    double gap;
    map_cuts[0] = root.p.cp;
    map_bound[0] = Bound(root_b);
    map_ub[0] = param.ub;
    time(&timer);
    queue.insert(std::pair<long, Node_DSDP>(0, root));

    while (!queue.empty()) {
        std::pair<long, double> next_kv = fetch_next<long, double>();
        long node_id = next_kv.first;
        Node_DSDP node = queue.at(node_id);
        long parent_id = node.id_parent;
//        // all finished
//        if (next_kv.second <= 1e-6) {
//            break;
//        }
        auto current_bound = map_bound[node_id];
        map_ub.erase(node_id);
        queue.erase(node_id);
        map_num_unsolved_child[node.id_parent] -= 1;
        int info = iter(node, param, qp);
        if (info == 1)
        {
            break;
        }
        else if (info == 0) {
            map_bound.erase(node_id);
            map_cuts.erase(node_id);
        }
        else {
            auto node_r = node.p.r;

            // branch
            Branch br(node_r);
            br.imply_bounds(current_bound);
            // create child
            long child_depth = node.depth + 1;
            long left_id = total_nodes + 1;
            long right_id = total_nodes + 2;

            // create cuts
            auto lc = CutPool();
            auto rc = CutPool();
            for (auto &c: node.p.cp) {
                lc.push_back(c);
                rc.push_back(c);
            }
            // todo, consider add more
            for(const auto &c: RLT_DSDP::create_from_branch(br, 0)){
                lc.push_back(c);
            }
            for(const auto &c: RLT_DSDP::create_from_branch(br, 1)){
                rc.push_back(c);
            }

            // nodes
            auto left_node = Node_DSDP(
                    left_id, qp, node_id, child_depth,
                    node.val_rel, 0.0, 0.0
            );
            auto right_node = Node_DSDP(
                    right_id, qp, node_id, child_depth,
                    node.val_rel, 0.0, 0.0
            );
            // update containers
            map_bound[left_id] = br.left_b;
            map_cuts[left_id] = lc;
            map_ub[left_id] = node.val_rel;
            map_num_cuts[left_id] = node.p.cp.size();
            queue.insert(std::pair<long, Node_DSDP>(left_id, left_node));
            map_bound[right_id] = br.right_b;
            map_cuts[right_id] = rc;
            map_ub[right_id] = node.val_rel;
            map_num_cuts[right_id] = node.p.cp.size();
            queue.insert(std::pair<long, Node_DSDP>(right_id, right_node));
            // add to result mapping temporarily
            map_result.insert(std::pair<long, Result_DSDP>(node_id, node_r));

            // counting
            total_nodes += 2;
            map_num_unsolved_child[node_id] = 2;
            // clean up
            map_bound.erase(node_id);
            map_cuts.erase(node_id);
        }
    }

    return 1;
}
//Node_DSDP Tree_DSDP::get_best_node() {
//    return queue.at(best_node);
//}

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> Tree_DSDP::fetch_next() {
    auto kv = get_max(map_ub);
    return kv;
}


